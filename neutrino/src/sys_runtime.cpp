// Opt-in whole-image CUDA interception. Upstream probe/replay paths are unused.
#include "sys_collect.hpp"
#include <cstdint>
using namespace neutrino::sys;
#undef cuGetProcAddress
#define NS_FAIL catch (const std::exception& error) { fatal(error.what()); }
#define NS_RUNTIME(name) runtime<decltype(&name)>(#name)

extern "C" void* dlsym(void* handle, const char* name) noexcept {
    void* original = raw_symbol(handle, name);
    if (original && enabled()) if (void* replacement = intercepted(name)) return replacement;
    return original;
}

extern "C" CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* path) {
    try {
        auto real = NS_DRIVER(cuModuleLoad); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(module, path);
        auto& s = state(); Lock lock(s.mutex); auto image = file_image(path);
        auto result = real(module, image && s.probes && image->probe_enabled ? image->manifest.at("replacement").get<std::string>().c_str() : path);
        if (result == CUDA_SUCCESS) s.modules[*module] = owner(image, path);
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* code) {
    try {
        auto real = NS_DRIVER(cuModuleLoadData); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(module, code);
        auto& s = state(); Lock lock(s.mutex); auto image = memory_image(code);
        auto result = real(module, image && s.probes && image->probe_enabled ? image->data.data() : code);
        if (result == CUDA_SUCCESS) s.modules[*module] = owner(image, "cuModuleLoadData");
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* code, unsigned count, CUjit_option* options, void** values) {
    try {
        auto real = NS_DRIVER(cuModuleLoadDataEx); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(module, code, count, options, values);
        auto& s = state(); Lock lock(s.mutex); auto image = memory_image(code);
        auto result = real(module, image && s.probes && image->probe_enabled ? image->data.data() : code, count, options, values);
        if (result == CUDA_SUCCESS) s.modules[*module] = owner(image, "cuModuleLoadDataEx");
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* code) {
    try {
        auto real = NS_DRIVER(cuModuleLoadFatBinary); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(module, code);
        auto& s = state(); Lock lock(s.mutex); auto image = memory_image(code);
        auto result = real(module, image && s.probes && image->probe_enabled ? image->data.data() : code);
        if (result == CUDA_SUCCESS) s.modules[*module] = owner(image, "cuModuleLoadFatBinary");
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuLibraryLoadFromFile(CUlibrary* library, const char* path,
        CUjit_option* options, void** values, unsigned count, CUlibraryOption* lib_options, void** lib_values, unsigned lib_count) {
    try {
        auto real = NS_DRIVER(cuLibraryLoadFromFile); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(library, path, options, values, count, lib_options, lib_values, lib_count);
        auto& s = state(); Lock lock(s.mutex); auto image = file_image(path);
        auto result = real(library, image && s.probes && image->probe_enabled ? image->manifest.at("replacement").get<std::string>().c_str() : path,
                           options, values, count, lib_options, lib_values, lib_count);
        if (result == CUDA_SUCCESS) s.libraries[*library] = owner(image, path);
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuLibraryLoadData(CUlibrary* library, const void* code,
        CUjit_option* options, void** values, unsigned count, CUlibraryOption* lib_options, void** lib_values, unsigned lib_count) {
    try {
        auto real = NS_DRIVER(cuLibraryLoadData); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(library, code, options, values, count, lib_options, lib_values, lib_count);
        auto& s = state(); Lock lock(s.mutex); auto image = memory_image(code);
        auto result = real(library, image && s.probes && image->probe_enabled ? image->data.data() : code, options, values, count, lib_options, lib_values, lib_count);
        if (result == CUDA_SUCCESS) s.libraries[*library] = owner(image, "cuLibraryLoadData");
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
    try {
        auto result = NS_DRIVER(cuModuleGetFunction)(function, module, name);
        if (enabled() && result == CUDA_SUCCESS) {
            auto& s = state(); Lock lock(s.mutex); auto found = s.modules.find(module);
            if (found != s.modules.end()) bind_function(*function, found->second, name);
        }
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuModuleEnumerateFunctions(CUfunction* functions, unsigned count, CUmodule module) {
    try {
        auto result = NS_DRIVER(cuModuleEnumerateFunctions)(functions, count, module);
        if (enabled() && result == CUDA_SUCCESS) {
            auto& s = state(); Lock lock(s.mutex); auto found = s.modules.find(module);
            if (found != s.modules.end()) {
                unsigned actual = 0; check(NS_DRIVER(cuModuleGetFunctionCount)(&actual, module));
                for (unsigned i = 0; i < std::min(actual, count); ++i) {
                    const char* name = nullptr; check(NS_DRIVER(cuFuncGetName)(&name, functions[i]));
                    bind_function(functions[i], found->second, name);
                }
            }
        }
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuKernelGetFunction(CUfunction* function, CUkernel kernel) {
    try {
        auto result = NS_DRIVER(cuKernelGetFunction)(function, kernel);
        if (enabled() && result == CUDA_SUCCESS) {
            auto& s = state(); Lock lock(s.mutex);
            CUlibrary library{}; check(NS_DRIVER(cuKernelGetLibrary)(&library, kernel));
            auto found = s.libraries.find(library);
            if (found != s.libraries.end()) {
                const char* name = nullptr; check(NS_DRIVER(cuKernelGetName)(&name, kernel));
                bind_function(*function, found->second, name);
            }
        }
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuLibraryGetModule(CUmodule* module, CUlibrary library) {
    try {
        auto result = NS_DRIVER(cuLibraryGetModule)(module, library);
        if (enabled() && result == CUDA_SUCCESS) {
            auto& s = state(); Lock lock(s.mutex); auto found = s.libraries.find(library);
            if (found != s.libraries.end()) s.modules[*module] = found->second;
        }
        return result;
    } NS_FAIL
}

static void retire(const std::shared_ptr<Owner>& value) {
    auto& s = state();
    for (const auto& item : s.pending) if (item.metadata.at("owner") == value->ordinal)
        throw ArgumentError("Image unloaded before capture drain");
    for (auto i = s.functions.begin(); i != s.functions.end();) {
        if (i->second.owner == value) i = s.functions.erase(i); else ++i;
    }
    for (auto i = s.host_functions.begin(); i != s.host_functions.end();) {
        if (i->second.owner == value) i = s.host_functions.erase(i); else ++i;
    }
    s.log({{"kind", "unload"}, {"owner", value->ordinal}});
}
extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    try {
        if (enabled()) { auto& s = state(); Lock lock(s.mutex); auto found = s.modules.find(module);
            if (found != s.modules.end()) { retire(found->second); s.modules.erase(found); } }
        return NS_DRIVER(cuModuleUnload)(module);
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuLibraryUnload(CUlibrary library) {
    try {
        if (enabled()) { auto& s = state(); Lock lock(s.mutex); auto found = s.libraries.find(library);
            if (found != s.libraries.end()) {
                retire(found->second);
                for (auto i = s.modules.begin(); i != s.modules.end();) {
                    if (i->second == found->second) i = s.modules.erase(i); else ++i;
                }
                s.libraries.erase(found);
            } }
        return NS_DRIVER(cuLibraryUnload)(library);
    } NS_FAIL
}

extern "C" void** __cudaRegisterFatBinary(void* data) {
    try {
        auto real = NS_RUNTIME(__cudaRegisterFatBinary); SubmissionScope scope;
        if (!enabled() || !scope.outermost()) return real(data);
        auto& s = state(); Lock lock(s.mutex);
        auto* wrapper = static_cast<__fatBinC_Wrapper_t*>(data);
        if (!wrapper || wrapper->magic != FATBINC_MAGIC || (wrapper->version != 1 && wrapper->version != 2))
            throw ArgumentError("Unsupported CUDA registration wrapper");
        auto image = memory_image(wrapper->data);
        auto value = owner(image, "__cudaRegisterFatBinary");
        if (image && s.probes && image->probe_enabled) {
            // The replacement is an already-linked whole image. Never retain
            // old-ABI relocatable fallbacks from a version-2 registration.
            value->wrapper = {FATBINC_MAGIC, FATBINC_VERSION, reinterpret_cast<const unsigned long long*>(image->data.data()), nullptr};
        }
        auto handle = real(image && s.probes && image->probe_enabled ? &value->wrapper : data);
        if (!handle) throw ArgumentError("CUDA registration failed");
        s.registrations[handle] = value; return handle;
    } NS_FAIL
}
extern "C" void __cudaRegisterFunction(void** registration, const char* host, char* device, const char* name,
                                      int limit, uint3* tid, uint3* bid, dim3* block, dim3* grid, int* warp) {
    try {
        auto real = NS_RUNTIME(__cudaRegisterFunction);
        if (enabled()) {
            auto& s = state(); Lock lock(s.mutex); auto found = s.registrations.find(registration);
            if (found == s.registrations.end()) throw ArgumentError("Unobserved CUDA registration handle");
            if (found->second->image && !found->second->image->entries.count(name)) throw ArgumentError("Registered entry layout missing");
            s.host_functions[host] = {found->second, name};
        }
        real(registration, host, device, name, limit, tid, bid, block, grid, warp);
    } NS_FAIL
}
extern "C" void __cudaUnregisterFatBinary(void** registration) {
    try {
        if (enabled()) { auto& s = state(); Lock lock(s.mutex); auto found = s.registrations.find(registration);
            if (found != s.registrations.end()) { retire(found->second); s.registrations.erase(found); } }
        NS_RUNTIME(__cudaUnregisterFatBinary)(registration);
    } NS_FAIL
}
extern "C" cudaError_t CUDARTAPI cudaGetFuncBySymbol(cudaFunction_t* function, const void* symbol) {
    try {
        auto result = NS_RUNTIME(cudaGetFuncBySymbol)(function, symbol);
        if (enabled() && result == cudaSuccess) {
            auto& s = state(); Lock lock(s.mutex); auto found = s.host_functions.find(symbol);
            if (found != s.host_functions.end()) bind_function(*function, found->second.owner, found->second.name.c_str());
        }
        return result;
    } NS_FAIL
}

static CUresult launch_driver(const char* name, const CUlaunchConfig& config, CUfunction function,
                              void** parameters, void** extra, bool extended, bool cooperative) {
    SubmissionScope scope;
    auto submit = [&](void** args, void** packed) {
        if (extended) return driver<decltype(&cuLaunchKernelEx)>(name)(&config, function, args, packed);
        if (cooperative) return driver<decltype(&cuLaunchCooperativeKernel)>(name)(function,
            config.gridDimX, config.gridDimY, config.gridDimZ, config.blockDimX, config.blockDimY, config.blockDimZ,
            config.sharedMemBytes, config.hStream, args);
        return driver<decltype(&cuLaunchKernel)>(name)(function,
            config.gridDimX, config.gridDimY, config.gridDimZ, config.blockDimX, config.blockDimY, config.blockDimZ,
            config.sharedMemBytes, config.hStream, args, packed);
    };
    if (!enabled() || !scope.outermost()) return submit(parameters, extra);
    return launch(function, nullptr, config, parameters, extra, name, cooperative, submit);
}
#define NS_LAUNCH(NAME) extern "C" CUresult CUDAAPI NAME(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** args, void** extra) { \
    try { CUlaunchConfig config{gx,gy,gz,bx,by,bz,shared,stream,nullptr,0}; return launch_driver(#NAME, config, function, args, extra, false, false); } NS_FAIL }
NS_LAUNCH(cuLaunchKernel)
NS_LAUNCH(cuLaunchKernel_ptsz)
#define NS_EXTENDED(NAME) extern "C" CUresult CUDAAPI NAME(const CUlaunchConfig* config, CUfunction function, void** args, void** extra) { \
    try { if (!config) return CUDA_ERROR_INVALID_VALUE; OwnedLaunchConfig owned(*config); return launch_driver(#NAME, *owned.get(), function, args, extra, true, false); } NS_FAIL }
NS_EXTENDED(cuLaunchKernelEx)
NS_EXTENDED(cuLaunchKernelEx_ptsz)
#define NS_COOPERATIVE(NAME) extern "C" CUresult CUDAAPI NAME(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream, void** args) { \
    try { CUlaunchConfig config{gx,gy,gz,bx,by,bz,shared,stream,nullptr,0}; return launch_driver(#NAME, config, function, args, nullptr, false, true); } NS_FAIL }
NS_COOPERATIVE(cuLaunchCooperativeKernel)
NS_COOPERATIVE(cuLaunchCooperativeKernel_ptsz)

static cudaError_t launch_runtime(const char* name, const void* host, dim3 grid, dim3 block, void** args,
                                  std::size_t shared, cudaStream_t stream, bool cooperative, const cudaLaunchConfig_t* extended = nullptr) {
    SubmissionScope scope;
    auto submit = [&](void** prepared, void**) {
        cudaError_t result;
        if (extended) result = runtime<decltype(&cudaLaunchKernelExC)>(name)(extended, host, prepared);
        else result = runtime<decltype(&cudaLaunchKernel)>(name)(host, grid, block, prepared, shared, stream);
        return static_cast<CUresult>(result);
    };
    if (!enabled() || !scope.outermost()) return static_cast<cudaError_t>(submit(args, nullptr));
    auto& s = state(); Lock lock(s.mutex);
    auto found = s.host_functions.find(host);
    Function descriptor = found == s.host_functions.end() ? Function{} : found->second;
    CUfunction function{};
    if (descriptor.owner && descriptor.owner->image) {
        const auto status = NS_RUNTIME(cudaGetFuncBySymbol)(&function, host);
        if (status != cudaSuccess) throw ArgumentError("Cannot resolve registered runtime kernel");
    }
    CUlaunchConfig config{grid.x,grid.y,grid.z,block.x,block.y,block.z,static_cast<unsigned>(shared),stream,nullptr,0};
    std::vector<CUlaunchAttribute> attrs;
    if (extended) {
        static_assert(sizeof(cudaLaunchAttribute) == sizeof(CUlaunchAttribute));
        attrs.resize(extended->numAttrs);
        if (!attrs.empty()) std::memcpy(attrs.data(), extended->attrs, attrs.size() * sizeof(CUlaunchAttribute));
        config.attrs = attrs.data(); config.numAttrs = attrs.size();
    }
    return static_cast<cudaError_t>(launch(function, &descriptor, config, args, nullptr, name, cooperative, submit));
}
#define NS_RT_LAUNCH(NAME, COOP) extern "C" cudaError_t CUDARTAPI NAME(const void* host, dim3 grid, dim3 block, void** args, std::size_t shared, cudaStream_t stream) { \
    try { return launch_runtime(#NAME, host, grid, block, args, shared, stream, COOP); } NS_FAIL }
NS_RT_LAUNCH(cudaLaunchKernel, false)
NS_RT_LAUNCH(cudaLaunchKernel_ptsz, false)
NS_RT_LAUNCH(cudaLaunchCooperativeKernel, true)
NS_RT_LAUNCH(cudaLaunchCooperativeKernel_ptsz, true)
NS_RT_LAUNCH(__cudaLaunchKernel, false)
NS_RT_LAUNCH(__cudaLaunchKernel_ptsz, false)
#define NS_RT_EXT(NAME) extern "C" cudaError_t CUDARTAPI NAME(const cudaLaunchConfig_t* config, const void* host, void** args) { \
    try { if (!config || (config->numAttrs && !config->attrs)) return cudaErrorInvalidValue; return launch_runtime(#NAME, host, config->gridDim, config->blockDim, args, config->dynamicSmemBytes, config->stream, false, config); } NS_FAIL }
NS_RT_EXT(cudaLaunchKernelExC)
NS_RT_EXT(cudaLaunchKernelExC_ptsz)

static void redirect(void** pointer) {
    // Match the pointer returned for the requested API version and stream mode,
    // not just the unversioned requested spelling.
    static const char* names[] = {"cuModuleLoad", "cuModuleLoadData", "cuModuleLoadDataEx", "cuModuleLoadFatBinary",
        "cuLibraryLoadFromFile", "cuLibraryLoadData", "cuModuleGetFunction", "cuModuleEnumerateFunctions", "cuKernelGetFunction",
        "cuLibraryGetModule", "cuModuleUnload", "cuLibraryUnload", "cuLaunchKernel", "cuLaunchKernel_ptsz",
        "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz", "cuLaunchCooperativeKernel", "cuLaunchCooperativeKernel_ptsz"};
    if (!enabled() || !pointer || !*pointer) return;
    for (const char* name : names) if (raw_symbol(driver_handle(), name) == *pointer) { *pointer = intercepted(name); return; }
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** pointer, int version, cuuint64_t flags) {
    try {
        auto result = driver<decltype(&cuGetProcAddress)>("cuGetProcAddress")(name, pointer, version, flags);
        if (result == CUDA_SUCCESS) redirect(pointer);
        return result;
    } NS_FAIL
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** pointer, int version, cuuint64_t flags, CUdriverProcAddressQueryResult* query) {
    try {
        auto result = driver<decltype(&cuGetProcAddress_v2)>("cuGetProcAddress_v2")(name, pointer, version, flags, query);
        if (result == CUDA_SUCCESS && (!query || *query == CU_GET_PROC_ADDRESS_SUCCESS)) redirect(pointer);
        return result;
    } NS_FAIL
}
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(const char* name, void** pointer, unsigned version,
                                                                 unsigned long long flags, cudaDriverEntryPointQueryResult* query) {
    try {
        auto result = NS_RUNTIME(cudaGetDriverEntryPointByVersion)(name, pointer, version, flags, query);
        if (result == cudaSuccess && (!query || *query == cudaDriverEntryPointSuccess)) redirect(pointer);
        return result;
    } NS_FAIL
}
extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char* name, void** pointer,
                                                        unsigned long long flags, cudaDriverEntryPointQueryResult* query) {
    try {
        auto result = NS_RUNTIME(cudaGetDriverEntryPoint)(name, pointer, flags, query);
        if (result == cudaSuccess && (!query || *query == cudaDriverEntryPointSuccess)) redirect(pointer);
        return result;
    } NS_FAIL
}

extern "C" std::uint32_t neutrinoSysAbiVersion() { return enabled() ? 1 : 0; }
extern "C" int neutrinoSysCaptureBegin(const char* name, std::int64_t ordinal) {
    try {
        if (!enabled()) return 1;
        auto& s = state(); Lock lock(s.mutex); SubmissionScope scope;
        if (s.capturing || s.failed || !s.pending.empty() || !name || ordinal < 0) throw ArgumentError("Invalid capture begin");
        auto& context = current_context();
        s.step = ordinal; initialize_capture(context); s.capturing = true;
        s.log({{"kind", "capture_begin"}, {"step", ordinal}, {"name", name}}); return 0;
    } NS_FAIL
}
extern "C" int neutrinoSysCaptureEnd(int successful) {
    try { auto& s = state(); Lock lock(s.mutex); SubmissionScope scope; end_capture(successful != 0); return 0; } NS_FAIL
}
extern "C" int neutrinoSysFinalize() {
    try { SubmissionScope scope; finish(); return 0; } NS_FAIL
}

namespace neutrino::sys {
void* intercepted(const char* name) {
#define NS_EXPORT(SYMBOL) if (std::strcmp(name, #SYMBOL) == 0) return reinterpret_cast<void*>(&SYMBOL)
    NS_EXPORT(cuModuleLoad); NS_EXPORT(cuModuleLoadData); NS_EXPORT(cuModuleLoadDataEx); NS_EXPORT(cuModuleLoadFatBinary);
    NS_EXPORT(cuLibraryLoadFromFile); NS_EXPORT(cuLibraryLoadData); NS_EXPORT(cuModuleGetFunction); NS_EXPORT(cuModuleEnumerateFunctions);
    NS_EXPORT(cuKernelGetFunction); NS_EXPORT(cuLibraryGetModule); NS_EXPORT(cuModuleUnload); NS_EXPORT(cuLibraryUnload);
    NS_EXPORT(cuLaunchKernel); NS_EXPORT(cuLaunchKernel_ptsz); NS_EXPORT(cuLaunchKernelEx); NS_EXPORT(cuLaunchKernelEx_ptsz);
    NS_EXPORT(cuLaunchCooperativeKernel); NS_EXPORT(cuLaunchCooperativeKernel_ptsz);
    NS_EXPORT(cuGetProcAddress); NS_EXPORT(cuGetProcAddress_v2);
    NS_EXPORT(cudaGetFuncBySymbol); NS_EXPORT(cudaGetDriverEntryPointByVersion); NS_EXPORT(cudaGetDriverEntryPoint);
    NS_EXPORT(cudaLaunchKernel); NS_EXPORT(cudaLaunchKernel_ptsz); NS_EXPORT(cudaLaunchKernelExC); NS_EXPORT(cudaLaunchKernelExC_ptsz);
    NS_EXPORT(cudaLaunchCooperativeKernel); NS_EXPORT(cudaLaunchCooperativeKernel_ptsz);
    NS_EXPORT(__cudaLaunchKernel); NS_EXPORT(__cudaLaunchKernel_ptsz);
    NS_EXPORT(__cudaRegisterFatBinary); NS_EXPORT(__cudaRegisterFunction); NS_EXPORT(__cudaUnregisterFatBinary);
#undef NS_EXPORT
    return nullptr;
}
} // namespace neutrino::sys
