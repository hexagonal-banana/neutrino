// Immutable image catalogs and owned native load/launch identities.
#pragma once
#include "sys_launch.hpp"
#include "sys_sampling.hpp"
#include <cuda_runtime_api.h>
#include <fatbinary_section.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <dlfcn.h>
#include <elf.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

extern char** environ;
namespace neutrino::sys {
using Json = nlohmann::json;
namespace fs = std::filesystem;
void* intercepted(const char* name);

inline void* raw_symbol(void* handle, const char* name) {
    static auto real = reinterpret_cast<void*(*)(void*, const char*)>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    if (!real) { std::fputs("Neutrino: cannot resolve glibc dlsym\n", stderr); std::abort(); }
    return real(handle, name);
}
inline std::string env(const char* key, const std::string& fallback = {}) {
    const char* value = std::getenv(key);
    return value && *value ? value : fallback;
}
inline bool enabled() { return !env("NEUTRINO_SYS_CATALOG").empty(); }
inline void* driver_handle() {
    static void* handle = [] {
        auto path = env("NEUTRINO_REAL_DRIVER");
        if (path.empty()) throw ArgumentError("An explicit real CUDA driver is required");
        void* result = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!result) throw ArgumentError("Cannot load real CUDA driver");
        return result;
    }();
    return handle;
}
template<class T> T driver(const char* name) {
    void* value = raw_symbol(driver_handle(), name);
    if (!value) throw ArgumentError(std::string("Missing CUDA driver API: ") + name);
    return reinterpret_cast<T>(value);
}
template<class T> T runtime(const char* name) {
    void* value = raw_symbol(RTLD_NEXT, name);
    if (!value) throw ArgumentError(std::string("Missing CUDA runtime API: ") + name);
    return reinterpret_cast<T>(value);
}
#define NS_DRIVER_EXPANDED(name) ::neutrino::sys::driver<decltype(&name)>(#name)
#define NS_DRIVER(name) NS_DRIVER_EXPANDED(name)
inline void check(CUresult status) {
    if (status != CUDA_SUCCESS) throw ArgumentError("CUDA status " + std::to_string(status));
}
inline std::uint64_t host_time() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value)) throw ArgumentError("Host clock failed");
    return std::uint64_t(value.tv_sec) * 1000000000ULL + value.tv_nsec;
}
inline std::string digest(const void* data, std::size_t size) {
    unsigned char bytes[SHA256_DIGEST_LENGTH];
    if (!SHA256(static_cast<const unsigned char*>(data), size, bytes)) throw ArgumentError("SHA256 failed");
    std::ostringstream out;
    for (auto byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}
inline std::vector<unsigned char> read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw ArgumentError("Cannot read image: " + path.string());
    const auto size = input.tellg();
    if (size <= 0 || size > 1024LL * 1024 * 1024) throw ArgumentError("Invalid image file size");
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    input.seekg(0); input.read(reinterpret_cast<char*>(data.data()), size);
    if (!input) throw ArgumentError("Incomplete image read");
    return data;
}
inline Json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw ArgumentError("Cannot read manifest: " + path.string());
    Json result; input >> result; return result;
}
inline std::size_t image_size(const void* ptr) {
    // CUDA requires valid image storage from the caller.
    if (!ptr) return 0;
    const auto* data = static_cast<const unsigned char*>(ptr);
    std::uint32_t magic; std::memcpy(&magic, ptr, 4);
    if (magic == 0xba55ed50) {
        std::uint16_t header; std::uint64_t payload;
        std::memcpy(&header, data + 6, 2); std::memcpy(&payload, data + 8, 8);
        if (header < 16 || header > 4096 || payload > (1ULL << 30)) throw ArgumentError("Invalid fatbin extent");
        return header + payload;
    }
    if (magic == 0x464c457f) {
        Elf64_Ehdr header; std::memcpy(&header, ptr, sizeof(header));
        if (header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
            !header.e_shnum || header.e_shentsize != sizeof(Elf64_Shdr)) return 0;
        std::uint64_t size = header.e_shoff + std::uint64_t(header.e_shnum) * header.e_shentsize;
        if (size > (1ULL << 30)) throw ArgumentError("Invalid ELF extent");
        for (unsigned i = 0; i < header.e_shnum; ++i) {
            Elf64_Shdr section;
            std::memcpy(&section, data + header.e_shoff + i * header.e_shentsize, sizeof(section));
            if (section.sh_type != SHT_NOBITS) size = std::max(size, section.sh_offset + section.sh_size);
        }
        if (size > (1ULL << 30)) throw ArgumentError("Invalid ELF section extent");
        return size;
    }
    if (magic == 0x7265762e || data[0] == '/' || data[0] == '\n') {
        const auto size = strnlen(reinterpret_cast<const char*>(data), 64 * 1024 * 1024);
        if (size == 64 * 1024 * 1024) throw ArgumentError("Unterminated PTX input");
        return size;
    }
    return 0;
}

inline int loop_depth(const Json& value) {
    const auto depth = value.value("loop_depth", Json(1));
    if (!depth.is_number_integer() || depth < 1 || depth > 2147483647 ||
        (value.at("interval_mode") != "loop-boundary" && depth != 1))
        throw ArgumentError("Invalid loop depth");
    return depth.get<int>();
}

inline int catalog_loop_depth(const Json& catalog, const std::string& library) {
    const int depth = loop_depth(catalog);
    const int jit_depth = loop_depth(Json{{"interval_mode", catalog.at("interval_mode")},
                                         {"loop_depth", catalog.value("jit_loop_depth", Json(depth))}});
    return library == "DeepEP" ? jit_depth : depth;
}

struct Image {
    Json manifest;
    bool probe_enabled = true;
    std::vector<unsigned char> data;
    std::map<std::string, std::shared_ptr<ParameterLayout>> entries;
    std::map<std::string, std::pair<std::string, int>> policies;
    explicit Image(Json value) : manifest(std::move(value)) {
        if (manifest.contains("probe_enabled") && !manifest.at("probe_enabled").is_boolean())
            throw ArgumentError("Image probe_enabled must be Boolean");
        probe_enabled = manifest.value("probe_enabled", true);
        if (manifest.at("interval_mode") != "instruction-only" && manifest.at("interval_mode") != "loop-boundary")
            throw ArgumentError("Unsupported interval mode");
        if (loop_depth(manifest) != loop_depth(manifest.at("probe")))
            throw ArgumentError("Image/probe loop depth mismatch");
        if (manifest.value("context_abi", 0) != SAMPLING_CONTEXT_ABI ||
            manifest.at("probe").value("context_abi", 0) != SAMPLING_CONTEXT_ABI ||
            manifest.at("probe").at("sampling") != "launch_policy")
            throw ArgumentError("Incompatible sampling context ABI");
        data = read_bytes(manifest.at("replacement").get<std::string>());
        if (digest(data.data(), data.size()) != manifest.at("replacement_sha256")) throw ArgumentError("Replacement changed");
        for (auto it = manifest.at("entries").begin(); it != manifest.at("entries").end(); ++it) {
            std::vector<Parameter> parameters;
            for (const auto& p : it.value().at("parameters")) parameters.push_back({p.at("offset"), p.at("size"), p.at("alignment")});
            const auto size = it.value().at("size").get<std::size_t>();
            entries.emplace(it.key(), std::make_shared<ParameterLayout>(parameters, size, 32764,
                                                                       it.value().value("packed_prefix_min", size)));
        }
    }
};
struct Owner {
    std::uint64_t ordinal;
    std::shared_ptr<Image> image;
    std::string source;
    __fatBinC_Wrapper_t wrapper{};
};
struct Function { std::shared_ptr<Owner> owner; std::string name; };
struct Pending {
    Json metadata;
    CUcontext context{};
    CUdeviceptr allocation{};
    std::size_t state_bytes{}, record_bytes{};
};
struct Context {
    CUcontext handle{};
    CUdevice device{};
    std::uint64_t ordinal{};
    std::string uuid;
    CUdeviceptr arena{}, host_device{};
    void* host_headers{};
    std::size_t budget{}, used{}, header_count{};
    CUmodule clock_module{};
    CUfunction clock_function{};
    CUdeviceptr clock_value{};
    CUstream clock_stream{};
};

struct State {
    std::recursive_mutex mutex;
    Json catalog, capture_policy;
    std::string capture_policy_hash;
    std::map<std::string, std::shared_ptr<Image>> images;
    std::map<CUmodule, std::shared_ptr<Owner>> modules;
    std::map<CUlibrary, std::shared_ptr<Owner>> libraries;
    std::map<void**, std::shared_ptr<Owner>> registrations;
    std::map<const void*, Function> host_functions;
    std::map<std::pair<CUcontext, CUfunction>, Function> functions;
    std::map<CUcontext, Context> contexts;
    std::vector<Pending> pending;
    std::ofstream journal;
    fs::path directory;
    std::uint64_t owners = 0, launches = 0, context_sequence = 0;
    std::int64_t step = -1;
    bool capturing = false, failed = false;
    std::string capture_mode;
    bool probes = true, cupti = true;
    unsigned capacity = 256;
    std::size_t budget = 2ULL * 1024 * 1024 * 1024;
    State() {
        capture_mode = env("NEUTRINO_SYS_CAPTURE_MODE", "full");
        if (capture_mode != "full" && capture_mode != "hooks" && capture_mode != "hooks-cupti")
            throw ArgumentError("Unsupported capture mode");
        probes = capture_mode == "full";
        cupti = capture_mode != "hooks";
        catalog = read_json(env("NEUTRINO_SYS_CATALOG"));
        if (catalog.at("schema") != 1 ||
            (catalog.at("interval_mode") != "instruction-only" && catalog.at("interval_mode") != "loop-boundary"))
            throw ArgumentError("Unsupported catalog");
        catalog_loop_depth(catalog, "DeepEP");
        if (catalog.value("context_abi", 0) != SAMPLING_CONTEXT_ABI)
            throw ArgumentError("Incompatible catalog context ABI");
        capture_policy = validate_policy(Json::parse(env("NEUTRINO_SYS_CAPTURE_POLICY", "{\"schema\":1}")));
        const auto canonical = capture_policy.dump();
        capture_policy_hash = digest(canonical.data(), canonical.size());
        if (env("NEUTRINO_SESSION_ID").empty() || env("NEUTRINO_SYS_OUTPUT").empty()) throw ArgumentError("Session/output required");
        directory = fs::path(env("NEUTRINO_SYS_OUTPUT")) / ("process-" + std::to_string(getpid()));
        if (!fs::create_directories(directory)) throw ArgumentError("Process output already exists");
        journal.open(directory / "journal.jsonl");
        if (!journal) throw ArgumentError("Cannot open trace journal");
        capacity = std::stoul(env("NEUTRINO_SYS_CAPACITY", "256"));
        budget = std::stoull(env("NEUTRINO_SYS_BUDGET", "2147483648"));
        if (!capacity || capacity > (1U << 20) || budget < 4096) throw ArgumentError("Invalid recording limits");
        for (const auto& image : catalog.at("images")) add_image(image);
        char hostname[256]{};
        std::string boot;
        std::ifstream boot_file("/proc/sys/kernel/random/boot_id"); boot_file >> boot;
        if (gethostname(hostname, sizeof(hostname) - 1) || boot.empty()) throw ArgumentError("Cannot identify host clock domain");
        log({{"kind", "process"}, {"session", env("NEUTRINO_SESSION_ID")}, {"rank", env("RANK", "0")},
             {"pid", getpid()}, {"started", host_time()}, {"capacity", capacity}, {"budget", budget},
             {"hostname", hostname}, {"boot_id", boot}, {"host_clock", "CLOCK_MONOTONIC_RAW"},
             {"context_abi", SAMPLING_CONTEXT_ABI}, {"capture_policy", capture_policy},
             {"capture_policy_sha256", capture_policy_hash}, {"capture_mode", capture_mode},
             {"kernel_probes", probes}, {"cupti_enabled", cupti}});
    }
    void log(Json value) {
        value["host_ns"] = host_time(); journal << value.dump() << '\n'; journal.flush();
        if (!journal) throw ArgumentError("Trace journal write failed");
    }
    std::shared_ptr<Image> add_image(const Json& value) {
        if (value.at("interval_mode") != catalog.at("interval_mode")) throw ArgumentError("Catalog/image interval mode mismatch");
        if (loop_depth(value) != catalog_loop_depth(catalog, value.at("label").get<std::string>()))
            throw ArgumentError("Catalog/image loop depth mismatch");
        auto image = std::make_shared<Image>(value);
        for (const auto& entry : image->entries)
            image->policies.emplace(entry.first, resolve_policy(capture_policy, entry.first, value.at("label")));
        for (const auto& hash : value.at("original_hashes")) {
            const auto key = hash.get<std::string>();
            auto found = images.find(key);
            if (found != images.end() && found->second->manifest != value) throw ArgumentError("Conflicting image identity");
            images[key] = image;
        }
        std::ofstream output(directory / (value.at("id").get<std::string>() + ".image.json"));
        output << value.dump(2); if (!output) throw ArgumentError("Image manifest write failed");
        return image;
    }
};
inline State& state() { static auto* instance = new State; return *instance; }
using Lock = std::lock_guard<std::recursive_mutex>;
[[noreturn]] inline void fatal(const std::string& message) {
    std::fprintf(stderr, "Neutrino sys_sync fatal (no launch retry): %s\n", message.c_str());
    try { auto& s = state(); Lock lock(s.mutex); s.failed = true; s.log({{"kind", "fatal"}, {"error", message}}); } catch (...) {}
    std::fflush(stderr); std::_Exit(86);
}
inline std::shared_ptr<Image> memory_image(const void* data) {
    const auto size = image_size(data);
    if (!size) return {};
    auto& images = state().images;
    auto found = images.find(digest(data, size));
    return found == images.end() ? nullptr : found->second;
}
inline std::shared_ptr<Image> file_image(const char* name) {
    auto bytes = read_bytes(name);
    auto& s = state();
    auto found = s.images.find(digest(bytes.data(), bytes.size()));
    if (found != s.images.end()) return found->second;
    const auto path = fs::canonical(name).string();
    bool selected = false;
    for (const auto& root : s.catalog.value("jit_roots", Json::array())) {
        const auto prefix = fs::canonical(root.get<std::string>()).string() + "/";
        selected |= path.compare(0, prefix.size(), prefix) == 0;
    }
    if (!selected) return {};
    if (!s.probes) throw ArgumentError("Observation-only JIT image must be present in the catalog: " + path);
    const auto python = env("NEUTRINO_SYS_PYTHON");
    const auto cache = env("NEUTRINO_SYS_CACHE");
    if (python.empty() || cache.empty()) throw ArgumentError("JIT resolver configuration missing");
    std::vector<std::string> args{python, "-m", "neutrino.sys_sync.runtime", "jit", "--image", path, "--cache", cache,
                                  "--interval-mode", s.catalog.at("interval_mode").get<std::string>(),
                                  "--loop-depth", std::to_string(catalog_loop_depth(s.catalog, "DeepEP"))};
    std::vector<char*> argv; for (auto& arg : args) argv.push_back(arg.data()); argv.push_back(nullptr);
    std::vector<std::string> strings;
    for (char** item = environ; *item; ++item) {
        std::string value(*item);
        if (value.rfind("LD_PRELOAD=", 0) && value.rfind("LD_AUDIT=", 0) &&
            value.rfind("CUDA_VISIBLE_DEVICES=", 0) && value.rfind("NEUTRINO_SYS_", 0)) strings.push_back(value);
    }
    strings.emplace_back("CUDA_VISIBLE_DEVICES=");
    std::vector<char*> environment; for (auto& value : strings) environment.push_back(value.data()); environment.push_back(nullptr);
    int pipes[2]; if (pipe(pipes)) throw ArgumentError("Resolver pipe failed");
    posix_spawn_file_actions_t actions; posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipes[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipes[0]);
    posix_spawn_file_actions_addclose(&actions, pipes[1]);
    pid_t pid; int status = posix_spawn(&pid, python.c_str(), &actions, nullptr, argv.data(), environment.data());
    posix_spawn_file_actions_destroy(&actions); close(pipes[1]);
    if (status) { close(pipes[0]); throw ArgumentError("Cannot spawn JIT resolver"); }
    std::string output; char chunk[4096]; ssize_t count;
    while ((count = read(pipes[0], chunk, sizeof(chunk))) > 0) {
        output.append(chunk, count); if (output.size() > 1024 * 1024) throw ArgumentError("Oversized resolver output");
    }
    close(pipes[0]);
    while (waitpid(pid, &status, 0) < 0) if (errno != EINTR) throw ArgumentError("Resolver wait failed");
    if (!WIFEXITED(status) || WEXITSTATUS(status)) throw ArgumentError("JIT resolver failed before image load: " + path);
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    auto image = s.add_image(read_json(output));
    if (image->manifest.at("id") != digest(bytes.data(), bytes.size())) throw ArgumentError("Resolver image identity mismatch");
    return image;
}
inline std::shared_ptr<Owner> owner(std::shared_ptr<Image> image, const std::string& source) {
    auto& s = state();
    auto result = std::make_shared<Owner>(); result->ordinal = ++s.owners; result->image = std::move(image); result->source = source;
    s.log({{"kind", "load"}, {"owner", result->ordinal}, {"source", source},
           {"image", result->image ? result->image->manifest.at("id") : Json(nullptr)},
           {"coverage", result->image ? (s.probes && result->image->probe_enabled ? "replaced" : "original") : "pass_through"}});
    return result;
}
inline void bind_function(CUfunction function, const std::shared_ptr<Owner>& value, const char* name) {
    if (!value) return;
    CUcontext context{}; check(NS_DRIVER(cuCtxGetCurrent)(&context));
    if (value->image && !value->image->entries.count(name)) throw ArgumentError("Entry absent from original layout manifest");
    state().functions[{context, function}] = {value, name};
}
} // namespace neutrino::sys
