// Per-capture recording arenas: no extra work is inserted between PDL kernels.
#pragma once
#include "sys_registry.hpp"
#include <atomic>
#include <cupti.h>

namespace neutrino::sys {
inline void activity_check(CUptiResult result) {
    if (result != CUPTI_SUCCESS) throw ArgumentError("CUPTI status " + std::to_string(result));
}
struct ActivityState {
    std::mutex mutex;
    bool initialized = false;
    CUpti_SubscriberHandle subscriber{};
    std::size_t callback_launches = 0;
    std::string error;
    std::size_t dropped = 0;
    std::map<std::uint32_t, std::uint64_t> correlations;
    std::vector<Json> kernels;
};
inline ActivityState& activity_state() { static auto* value = new ActivityState; return *value; }
inline void record_correlation(ActivityState& activity, std::uint32_t correlation, std::uint64_t ordinal) {
    auto inserted = activity.correlations.emplace(correlation, ordinal);
    if (!inserted.second && inserted.first->second != ordinal)
        throw ArgumentError("Conflicting CUPTI launch correlation");
}
inline void CUPTIAPI activity_launch_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId, const void* raw) {
    const auto ordinal = ActivityCorrelationScope::current();
    if (!ordinal || domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const auto* info = static_cast<const CUpti_CallbackData*>(raw);
    if (info->callbackSite != CUPTI_API_ENTER || !info->functionName) return;
    const char* name = info->functionName;
    if (std::strcmp(name, "cuLaunchKernel") && std::strcmp(name, "cuLaunchKernel_ptsz") &&
        std::strcmp(name, "cuLaunchKernelEx") && std::strcmp(name, "cuLaunchKernelEx_ptsz") &&
        std::strcmp(name, "cuLaunchCooperativeKernel") && std::strcmp(name, "cuLaunchCooperativeKernel_ptsz")) return;
    // Nested runtime->driver calls can omit external-correlation activity records.
    // The callback and kernel activity share this exact driver correlation ID.
    // Never call CUDA, rewrite arguments, or submit work from this observer.
    auto& activity = activity_state(); std::lock_guard<std::mutex> lock(activity.mutex);
    try {
        record_correlation(activity, info->correlationId, ordinal);
        ++activity.callback_launches;
    } catch (const std::exception& error) { activity.error = error.what(); }
}
inline std::uint64_t CUPTIAPI activity_timestamp() { return host_time(); }
inline void CUPTIAPI activity_request(std::uint8_t** buffer, std::size_t* bytes, std::size_t* records) {
    *bytes = 8 * 1024 * 1024; *records = 0;
    *buffer = static_cast<std::uint8_t*>(std::malloc(*bytes));
    if (!*buffer) std::_Exit(86);
}
inline void CUPTIAPI activity_complete(CUcontext context, std::uint32_t stream,
                                      std::uint8_t* buffer, std::size_t, std::size_t valid) {
    auto& a = activity_state(); std::lock_guard<std::mutex> lock(a.mutex);
    try {
        CUpti_Activity* record = nullptr;
        for (;;) {
            auto status = cuptiActivityGetNextRecord(buffer, valid, &record);
            if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            activity_check(status);
            if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
                auto* r = reinterpret_cast<CUpti_ActivityExternalCorrelation*>(record);
                if (r->externalKind != CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) continue;
                record_correlation(a, r->correlationId, r->externalId);
            } else if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                auto* r = reinterpret_cast<CUpti_ActivityKernel11*>(record);
                a.kernels.push_back({{"kind", "kernel_activity"}, {"correlation", +r->correlationId},
                    {"begin_host_ns", +r->start}, {"end_host_ns", +r->end}, {"kernel", std::string(r->name)},
                    {"device", +r->deviceId}, {"cupti_context", +r->contextId}, {"cupti_stream", +r->streamId},
                    {"grid", {+r->gridX, +r->gridY, +r->gridZ}}, {"block", {+r->blockX, +r->blockY, +r->blockZ}},
                    {"source", "CUPTI_CONCURRENT_KERNEL"}, {"host_clock", "CLOCK_MONOTONIC_RAW"}});
            }
        }
        std::size_t dropped = 0;
        activity_check(cuptiActivityGetNumDroppedRecords(context, stream, &dropped));
        a.dropped += dropped;
    } catch (const std::exception& e) { a.error = e.what(); }
    std::free(buffer);
}
inline void activity_flush(bool emit) {
    auto& a = activity_state();
    activity_check(cuptiActivityFlushAll(0));
    std::lock_guard<std::mutex> lock(a.mutex);
    if (!a.error.empty() || a.dropped) throw ArgumentError("Incomplete CUPTI activity: " + a.error);
    if (emit) {
        auto& s = state();
        std::map<std::uint64_t, const Pending*> expected;
        for (const auto& pending : s.pending) expected.emplace(pending.metadata.at("ordinal"), &pending);
        std::size_t matched = 0;
        for (auto& kernel : a.kernels) {
            auto correlation = a.correlations.find(kernel.at("correlation"));
            if (correlation == a.correlations.end()) continue;
            auto found = expected.find(correlation->second);
            if (found == expected.end()) throw ArgumentError("Orphan or duplicate CUPTI launch");
            const auto& metadata = found->second->metadata;
            if (kernel.at("begin_host_ns") == 0 || kernel.at("end_host_ns") <= kernel.at("begin_host_ns") ||
                kernel.at("grid") != metadata.at("grid") || kernel.at("block") != metadata.at("block") ||
                kernel.at("device") != s.contexts.at(found->second->context).device)
                throw ArgumentError("CUPTI kernel identity/timing mismatch");
            kernel["ordinal"] = correlation->second; kernel["step"] = s.step;
            s.log(kernel); expected.erase(found); ++matched;
        }
        if (!expected.empty()) {
            Json missing = Json::array(), candidates = Json::array();
            for (const auto& item : expected) missing.push_back(item.second->metadata);
            for (const auto& kernel : a.kernels)
                if (std::any_of(expected.begin(), expected.end(), [&](const auto& item) {
                    return item.second->metadata.at("kernel") == kernel.at("kernel");
                })) candidates.push_back(kernel);
            s.log({{"kind", "activity_coverage_error"}, {"step", s.step}, {"missing", missing},
                   {"candidate_kernels", candidates}, {"correlations", a.correlations}});
            throw ArgumentError("Selected launch has no independent CUPTI kernel activity");
        }
        s.log({{"kind", "activity_validation"}, {"step", s.step}, {"matched", matched}, {"dropped", a.dropped},
               {"driver_callback_launches", a.callback_launches}});
    }
    a.correlations.clear(); a.kernels.clear(); a.callback_launches = 0;
}
inline void initialize_activity() {
    auto& a = activity_state();
    if (!a.initialized) {
        activity_check(cuptiSubscribe(&a.subscriber, activity_launch_callback, nullptr));
        activity_check(cuptiEnableDomain(1, a.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
        activity_check(cuptiActivityRegisterTimestampCallback(activity_timestamp));
        activity_check(cuptiActivityRegisterCallbacks(activity_request, activity_complete));
        activity_check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
        activity_check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
        activity_check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
        activity_check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        a.initialized = true;
    }
    activity_flush(false);
}
constexpr std::size_t HEADER_SLOTS = 65536;
struct TraceContext {
    std::uint64_t records, states, launch, threads;
    std::uint32_t capacity, schema;
    std::uint64_t reserved, sampling;
};
static_assert(sizeof(TraceContext) == 56);
static_assert(offsetof(TraceContext, sampling) == 48);
constexpr const char* CLOCK_PTX = R"(.version 9.1
.target sm_90
.address_size 64
.visible .entry neutrino_clock(.param .u64 out) {
.reg .u64 %p, %t;
ld.param.u64 %p, [out];
mov.u64 %t, %globaltimer;
st.global.u64 [%p], %t;
ret;
})";
void finish();
inline Context& current_context() {
    auto& s = state();
    CUcontext handle{}; check(NS_DRIVER(cuCtxGetCurrent)(&handle));
    if (!handle) throw ArgumentError("Launch has no current CUDA context");
    auto found = s.contexts.find(handle);
    if (found != s.contexts.end()) return found->second;
    Context context; context.handle = handle; context.ordinal = ++s.context_sequence;
    check(NS_DRIVER(cuCtxGetDevice)(&context.device));
    int major = 0, minor = 0;
    check(NS_DRIVER(cuDeviceGetAttribute)(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, context.device));
    check(NS_DRIVER(cuDeviceGetAttribute)(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, context.device));
    if (major != 9 || minor != 0) throw ArgumentError("This catalog requires Hopper 9.0; no architecture retargeting");
    CUuuid uuid{}; check(NS_DRIVER(cuDeviceGetUuid)(&uuid, context.device));
    std::ostringstream text; text << "GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) text << '-';
        text << std::hex << std::setw(2) << std::setfill('0') << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
    }
    context.uuid = text.str();
    s.log({{"kind", "context"}, {"context", context.ordinal}, {"uuid", context.uuid}, {"device", context.device}});
    return s.contexts.emplace(handle, context).first->second;
}
inline void clock_samples(Context& context, const char* phase) {
    if (!context.clock_module) {
        check(NS_DRIVER(cuModuleLoadData)(&context.clock_module, CLOCK_PTX));
        check(NS_DRIVER(cuModuleGetFunction)(&context.clock_function, context.clock_module, "neutrino_clock"));
        check(NS_DRIVER(cuMemAlloc)(&context.clock_value, sizeof(std::uint64_t)));
        check(NS_DRIVER(cuStreamCreate)(&context.clock_stream, CU_STREAM_NON_BLOCKING));
    }
    Json samples = Json::array();
    for (int i = -4; i < 32; ++i) {
        std::uint64_t tick = 0; void* args[]{&context.clock_value};
        auto before = host_time();
        check(NS_DRIVER(cuLaunchKernel)(context.clock_function, 1, 1, 1, 1, 1, 1, 0, context.clock_stream, args, nullptr));
        check(NS_DRIVER(cuStreamSynchronize)(context.clock_stream));
        check(NS_DRIVER(cuMemcpyDtoH)(&tick, context.clock_value, sizeof(tick)));
        auto after = host_time();
        if (i >= 0) samples.push_back({before, tick, after});
    }
    state().log({{"kind", "calibration"}, {"step", state().step}, {"context", context.ordinal},
                 {"uuid", context.uuid}, {"phase", phase}, {"samples", samples}});
}
inline void initialize_capture(Context& context) {
    auto& s = state();
    if (!s.probes) {
        if (s.cupti) initialize_activity();
        return;
    }
    if (!context.arena) {
        context.budget = s.budget;
        check(NS_DRIVER(cuMemAlloc)(&context.arena, context.budget));
        check(NS_DRIVER(cuMemHostAlloc)(&context.host_headers, HEADER_SLOTS * sizeof(TraceContext), CU_MEMHOSTALLOC_DEVICEMAP));
        check(NS_DRIVER(cuMemHostGetDevicePointer)(&context.host_device, context.host_headers, 0));
        static bool registered = false;
        if (!registered) { std::atexit([] { try { finish(); } catch (const std::exception& e) { fatal(e.what()); } }); registered = true; }
    }
    context.used = context.header_count = 0;
    check(NS_DRIVER(cuMemsetD8)(context.arena, 0, context.budget));
    initialize_activity();
    clock_samples(context, "before");
}
inline void release_context(Context& context) {
    if (context.arena) check(NS_DRIVER(cuMemFree)(context.arena));
    if (context.host_headers) check(NS_DRIVER(cuMemFreeHost)(context.host_headers));
    if (context.clock_value) check(NS_DRIVER(cuMemFree)(context.clock_value));
    if (context.clock_stream) check(NS_DRIVER(cuStreamDestroy)(context.clock_stream));
    if (context.clock_module) check(NS_DRIVER(cuModuleUnload)(context.clock_module));
    context.arena = context.clock_value = context.host_device = 0;
    context.host_headers = nullptr; context.clock_stream = nullptr; context.clock_module = nullptr;
}
inline void finish() {
    if (!enabled()) return;
    auto& s = state(); Lock lock(s.mutex);
    if (s.failed) return;
    if (s.capturing || !s.pending.empty()) throw ArgumentError("Process exited with an incomplete capture range");
    for (auto& pair : s.contexts) {
        auto& context = pair.second;
        if (!context.arena && !context.clock_module) continue;
        check(NS_DRIVER(cuCtxPushCurrent)(context.handle));
        release_context(context);
        CUcontext popped{}; check(NS_DRIVER(cuCtxPopCurrent)(&popped));
    }
    s.log({{"kind", "finalized"}, {"launches", s.launches}});
}
inline std::uint64_t product(unsigned x, unsigned y, unsigned z) {
    std::uint64_t value = x;
    if (!x || !y || !z || value > UINT64_MAX / y || (value *= y) > UINT64_MAX / z)
        throw ArgumentError("Invalid launch geometry");
    return value * z;
}
inline void residency(CUfunction function, const CUlaunchConfig& config, bool cooperative) {
    for (unsigned i = 0; i < config.numAttrs; ++i)
        if (config.attrs[i].id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE) cooperative |= config.attrs[i].value.cooperative != 0;
    if (!cooperative) return;
    const auto block = product(config.blockDimX, config.blockDimY, config.blockDimZ);
    int active = 0, sms = 0;
    check(NS_DRIVER(cuOccupancyMaxActiveBlocksPerMultiprocessor)(&active, function, static_cast<int>(block), config.sharedMemBytes));
    check(NS_DRIVER(cuDeviceGetAttribute)(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, current_context().device));
    if (product(config.gridDimX, config.gridDimY, config.gridDimZ) > std::uint64_t(active) * sms)
        throw ArgumentError("Instrumented cooperative grid does not fit; refusing launch");
}

struct LaunchContextBinding {
    bool pushed = false;
    CUcontext handle{};
    explicit LaunchContextBinding(CUstream stream) {
        check(NS_DRIVER(cuCtxGetCurrent)(&handle));
        if (!handle && stream && stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD) {
            check(NS_DRIVER(cuStreamGetCtx)(stream, &handle));
            if (handle) { check(NS_DRIVER(cuCtxPushCurrent)(handle)); pushed = true; }
        }
    }
    ~LaunchContextBinding() {
        if (pushed) {
            CUcontext popped{};
            if (NS_DRIVER(cuCtxPopCurrent)(&popped) != CUDA_SUCCESS || popped != handle)
                fatal("Cannot restore launch thread context");
        }
    }
};

template<class Submit> CUresult launch(CUfunction function, const Function* known, const CUlaunchConfig& config,
                                     void** parameters, void** extra, const char* api, bool cooperative, Submit submit) {
    auto& s = state(); Lock lock(s.mutex);
    LaunchContextBinding binding(config.hStream);
    if (!binding.handle) {
        // Runtime kernels on a fresh host thread can initialize their implicit
        // context inside the original API. Do not reject unrelated CUDA work.
        bool selected = known && known->owner && known->owner->image;
        if (function) for (const auto& item : s.functions)
            selected |= item.first.second == function && bool(item.second.owner->image);
        if (selected) throw ArgumentError(std::string("Selected launch has no resolvable context: ") + api);
        const auto ordinal = ++s.launches;
        const auto status = submit(parameters, extra);
        s.log({{"kind", "launch"}, {"ordinal", ordinal}, {"step", s.capturing ? s.step : -1},
               {"api", api}, {"context", nullptr}, {"uuid", nullptr}, {"status", status},
               {"kernel", known ? known->name : ""}, {"coverage", "pass_through"},
               {"reason", "implicit_runtime_context"}});
        return status;
    }
    auto& context = current_context();
    Function descriptor;
    if (known) descriptor = *known;
    else {
        auto found = s.functions.find({context.handle, function});
        if (found != s.functions.end()) descriptor = found->second;
    }
    const auto ordinal = ++s.launches;
    auto image = descriptor.owner ? descriptor.owner->image : nullptr;
    Json metadata{{"kind", "launch"}, {"ordinal", ordinal}, {"step", s.capturing ? s.step : -1},
                  {"api", api}, {"context", context.ordinal}, {"uuid", context.uuid},
                  {"stream", reinterpret_cast<std::uintptr_t>(config.hStream)}, {"kernel", descriptor.name},
                  {"grid", {config.gridDimX, config.gridDimY, config.gridDimZ}},
                  {"block", {config.blockDimX, config.blockDimY, config.blockDimZ}}, {"shared", config.sharedMemBytes},
                  {"owner", descriptor.owner ? descriptor.owner->ordinal : 0},
                  {"image", image ? image->manifest.at("id") : Json(nullptr)},
                  {"coverage", image ? (s.probes && image->probe_enabled ? "instrumented" : "observed") : "pass_through"}};
    if (!image) {
        const auto status = submit(parameters, extra);
        metadata["status"] = status; s.log(metadata); return status;
    }
    const auto& policy = image->policies.at(descriptor.name);
    const auto sampling = sampling_mode(policy.first);
    metadata["capture_sampling"] = policy.first;
    metadata["capture_policy_rule"] = policy.second;
    metadata["capture_policy_sha256"] = s.capture_policy_hash;
    metadata["context_abi"] = SAMPLING_CONTEXT_ABI;
    if (function) residency(function, config, cooperative);
    CUstreamCaptureStatus graph{};
    check(NS_DRIVER(cuStreamIsCapturing)(config.hStream, &graph));
    if (graph != CU_STREAM_CAPTURE_STATUS_NONE) throw ArgumentError("CUDA Graph capture of replaced images is unsupported");
    if (!s.probes || !image->probe_enabled) {
        // Original images and argument ABI: no hidden context or inserted GPU code.
        if (s.capturing && s.cupti)
            activity_check(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, ordinal));
        const auto status = [&] {
            ActivityCorrelationScope correlation(s.capturing && s.cupti ? ordinal : 0);
            return submit(parameters, extra);
        }();
        if (s.capturing && s.cupti) {
            std::uint64_t popped = 0;
            activity_check(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped));
            if (popped != ordinal) throw ArgumentError("CUPTI external correlation stack mismatch");
        }
        metadata["status"] = status; s.log(metadata);
        if (status != CUDA_SUCCESS) throw ArgumentError("Observed launch failed; it will not be retried");
        if (s.capturing) {
            Pending pending; pending.metadata = metadata; pending.context = context.handle;
            s.pending.push_back(std::move(pending));
        }
        return status;
    }
    std::uint64_t device_context = 0;
    Pending pending;
    if (s.capturing) {
        if (!context.arena) throw ArgumentError("New CUDA context appeared inside capture; initialize it before begin");
        const auto blocks = product(config.gridDimX, config.gridDimY, config.gridDimZ);
        const auto width = product(config.blockDimX, config.blockDimY, config.blockDimZ);
        if (blocks > UINT64_MAX / width) throw ArgumentError("Thread count overflow");
        const auto logical_threads = blocks * width;
        const auto threads = blocks * sampled_width(width, sampling);
        const bool pooled = image->manifest.at("probe").value("record_storage", "per-thread-v1") == "pooled-v1";
        unsigned capacity = image->manifest.at("probe").at("sites").empty() ? 1 : s.capacity;
        if (threads > context.budget / (16ULL + (pooled ? 40ULL : 32ULL) * capacity)) throw ArgumentError("Single-launch trace exceeds budget");
        pending.state_bytes = threads * 16;
        if (pooled) {
            if (threads * capacity > UINT32_MAX) throw ArgumentError("Pool slot count exceeds nonwrapping allocator domain");
            capacity = static_cast<unsigned>(threads * capacity);
            pending.record_bytes = std::size_t(capacity) * 40 + 8;
        } else pending.record_bytes = threads * capacity * 32;
        const auto bytes = pending.state_bytes + pending.record_bytes;
        if (bytes > context.budget || context.used > context.budget - bytes || context.header_count == HEADER_SLOTS) throw ArgumentError("Trace arena exhausted");
        pending.allocation = context.arena + context.used; pending.context = context.handle;
        const auto counter = pending.allocation + pending.state_bytes;
        TraceContext header{counter + (pooled ? 8 : 0), pending.allocation, ordinal, threads, capacity,
                            pooled ? 4U : 3U, pooled ? counter : 0, sampling};
        const auto offset = context.header_count++ * sizeof(header);
        std::memcpy(static_cast<unsigned char*>(context.host_headers) + offset, &header, sizeof(header));
        std::atomic_thread_fence(std::memory_order_seq_cst);
        device_context = context.host_device + offset;
        context.used += bytes;
        metadata["threads"] = threads; metadata["capacity"] = capacity;
        metadata["logical_threads"] = logical_threads; metadata["recorded_threads"] = threads;
        metadata["raw"] = "launch-" + std::to_string(ordinal) + ".bin";
        metadata["raw_format"] = "states-packed-v2";
        metadata["record_storage"] = pooled ? "pooled-v1" : "per-thread-v1";
    }
    PreparedArguments args(*image->entries.at(descriptor.name), parameters, extra, device_context);
    // Original handles/configuration are retained. Never replay a submitted call.
    if (s.capturing) activity_check(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, ordinal));
    const auto status = [&] {
        ActivityCorrelationScope correlation(s.capturing ? ordinal : 0);
        return submit(args.parameters(), args.extra());
    }();
    if (s.capturing) {
        std::uint64_t popped = 0;
        activity_check(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped));
        if (popped != ordinal) throw ArgumentError("CUPTI external correlation stack mismatch");
    }
    metadata["status"] = status; s.log(metadata);
    if (status != CUDA_SUCCESS) throw ArgumentError("Replaced launch failed; it will not be retried");
    if (s.capturing) { pending.metadata = metadata; s.pending.push_back(std::move(pending)); }
    return status;
}
inline void end_capture(bool successful) {
    auto& s = state();
    if (!s.capturing) throw ArgumentError("No active capture range");
    if (!s.probes) {
        for (auto& pair : s.contexts) {
            check(NS_DRIVER(cuCtxPushCurrent)(pair.second.handle));
            check(NS_DRIVER(cuCtxSynchronize)());
            CUcontext popped{}; check(NS_DRIVER(cuCtxPopCurrent)(&popped));
        }
        if (s.cupti) activity_flush(true);
        s.log({{"kind", "capture_end"}, {"step", s.step}, {"successful", successful},
               {"recorded_launches", 0}, {"observed_launches", s.pending.size()}});
        s.pending.clear(); s.capturing = false;
        if (!successful) throw ArgumentError("Observation-only capture failed");
        return;
    }
    for (auto& pair : s.contexts) {
        auto& context = pair.second;
        check(NS_DRIVER(cuCtxPushCurrent)(context.handle));
        check(NS_DRIVER(cuCtxSynchronize)());
        for (const auto& pending : s.pending) {
            if (pending.context != context.handle || !pending.metadata.contains("raw")) continue;
            auto data = std::vector<unsigned char>(pending.state_bytes);
            check(NS_DRIVER(cuMemcpyDtoH)(data.data(), pending.allocation, data.size()));
            std::ofstream output(s.directory / pending.metadata.at("raw").get<std::string>(), std::ios::binary);
            output.write(reinterpret_cast<const char*>(data.data()), data.size());
            if (!output) throw ArgumentError("Raw record write failed");
            // Overflow is evidence, not permission to return a truncated trace.
            std::size_t overflow = 0, incomplete = 0;
            std::uint32_t maximum = 0;
            const unsigned capacity = pending.metadata.at("capacity");
            for (std::size_t offset = 0; offset < pending.state_bytes; offset += 16) {
                std::uint32_t fields[4]; std::memcpy(fields, data.data() + offset, 16);
                overflow += fields[1] != 0;
                incomplete += fields[2] != 1 || fields[3] || fields[0] > capacity;
                maximum = std::max(maximum, fields[0]);
            }
            // Persist every state followed by used records in thread/ordinal order.
            // Bound host staging memory and omit untouched device record regions.
            const std::size_t threads = pending.state_bytes / 16;
            const std::size_t stride = std::size_t(capacity) * 32;
            const std::size_t batch = std::max<std::size_t>(1, (8 * 1024 * 1024) / stride);
            std::uint64_t records = 0;
            if (pending.metadata.value("record_storage", "per-thread-v1") == "pooled-v1") {
                std::uint32_t slots = 0;
                check(NS_DRIVER(cuMemcpyDtoH)(&slots, pending.allocation + pending.state_bytes, sizeof(slots)));
                if (slots > capacity) throw ArgumentError("Pool allocator exceeded capacity");
                std::vector<std::vector<std::array<unsigned char, 32>>> bins(threads);
                constexpr std::size_t chunk_slots = 32768;
                for (std::size_t first = 0; first < slots; first += chunk_slots) {
                    const auto count = std::min<std::size_t>(chunk_slots, slots - first);
                    std::vector<unsigned char> chunk(count * 40);
                    check(NS_DRIVER(cuMemcpyDtoH)(chunk.data(), pending.allocation + pending.state_bytes + 8 + first * 40, chunk.size()));
                    for (std::size_t slot = 0; slot < count; ++slot) {
                        const auto* record = chunk.data() + slot * 40;
                        std::uint64_t thread; std::uint32_t ordinal;
                        std::memcpy(&thread, record + 32, 8); std::memcpy(&ordinal, record + 24, 4);
                        const auto& shape = pending.metadata.at("block");
                        const auto width = product(shape[0], shape[1], shape[2]);
                        if (thread >= pending.metadata.at("logical_threads").get<std::uint64_t>())
                            throw ArgumentError("Pool physical thread outside launch");
                        thread = sampled_slot(thread, width, sampling_mode(pending.metadata.at("capture_sampling")));
                        if (thread >= threads || ordinal != bins[thread].size()) throw ArgumentError("Orphan/duplicate pool record");
                        std::array<unsigned char, 32> value; std::memcpy(value.data(), record, 32);
                        bins[thread].push_back(value);
                    }
                }
                for (std::size_t thread = 0; thread < threads; ++thread) {
                    std::uint32_t used; std::memcpy(&used, data.data() + thread * 16, 4);
                    if (used != bins[thread].size()) throw ArgumentError("Pool records disagree with thread state");
                    if (used) output.write(reinterpret_cast<const char*>(bins[thread].data()), std::size_t(used) * 32);
                    records += used;
                }
            } else for (std::size_t first = 0; first < threads; first += batch) {
                const auto last = std::min(threads, first + batch);
                std::size_t extent = 0;
                for (auto thread = first; thread < last; ++thread) {
                    std::uint32_t used; std::memcpy(&used, data.data() + thread * 16, 4);
                    if (used) extent = (thread - first) * stride + std::size_t(std::min(used, capacity)) * 32;
                }
                if (!extent) continue;
                std::vector<unsigned char> chunk(extent);
                check(NS_DRIVER(cuMemcpyDtoH)(chunk.data(), pending.allocation + pending.state_bytes + first * stride, extent));
                for (auto thread = first; thread < last; ++thread) {
                    std::uint32_t used; std::memcpy(&used, data.data() + thread * 16, 4);
                    used = std::min(used, capacity);
                    if (used) output.write(reinterpret_cast<const char*>(chunk.data() + (thread - first) * stride), std::size_t(used) * 32);
                    records += used;
                }
            }
            output.close();
            if (!output) throw ArgumentError("Compact record write failed");
            s.log({{"kind", "drain"}, {"ordinal", pending.metadata.at("ordinal")},
                   {"maximum_used", maximum}, {"overflow_threads", overflow}, {"incomplete_threads", incomplete},
                   {"records", records}, {"raw_bytes", pending.state_bytes + records * 32}});
            if (overflow || incomplete) successful = false;
        }
        if (context.arena) clock_samples(context, "after");
        CUcontext popped{}; check(NS_DRIVER(cuCtxPopCurrent)(&popped));
    }
    activity_flush(true);
    std::size_t recorded = 0;
    for (const auto& pending : s.pending) recorded += pending.metadata.contains("raw");
    s.log({{"kind", "capture_end"}, {"step", s.step}, {"successful", successful},
           {"recorded_launches", recorded}, {"observed_launches", s.pending.size() - recorded}});
    s.pending.clear(); s.capturing = false;
    if (!successful) throw ArgumentError("Capture failed, overflowed, or contains incomplete thread states");
}
} // namespace neutrino::sys
