/* Fixed-slot, single-execution DeepJIT tracing. Included by cuda.c. */
#include <spawn.h>
#include <inttypes.h>
extern char **environ;
static void init(void);
static pthread_mutex_t nb_mutex = PTHREAD_MUTEX_INITIALIZER;
static int nb_failed, nb_exit_registered;
static void nb_shutdown(void);
static int nb_mode(void) { const char *s = getenv("NEUTRINO_BARRIER_MODE"); return s && !strcmp(s, "1"); }
static int nb_sync_mode(void) { const char *s = getenv("NEUTRINO_PTX_SYNC"); return s && !strcmp(s, "1"); }
#define NB_API(name) __typeof__(&name) nb_##name
static NB_API(cuLibraryLoadFromFile);
static NB_API(cuLibraryEnumerateKernels);
static NB_API(cuLibraryUnload);
static NB_API(cuKernelGetName);
static NB_API(cuFuncGetName);
static NB_API(cuCtxGetCurrent);
static NB_API(cuCtxGetDevice);
static NB_API(cuCtxPushCurrent_v2);
static NB_API(cuCtxPopCurrent_v2);
static NB_API(cuDeviceGetUuid);
static NB_API(cuDeviceGetAttribute);
static NB_API(cuEventCreate);
static NB_API(cuEventRecord);
static NB_API(cuEventQuery);
static NB_API(cuEventSynchronize);
static NB_API(cuEventDestroy_v2);
static NB_API(cuMemHostAlloc);
static NB_API(cuMemFreeHost);
static NB_API(cuMemsetD8Async);
static NB_API(cuMemcpyDtoHAsync_v2);
static NB_API(cuFuncGetAttribute);
static NB_API(cuFuncSetAttribute);
static NB_API(cuOccupancyMaxActiveBlocksPerMultiprocessor);
static NB_API(cuStreamIsCapturing);
static NB_API(cuModuleUnload);
static NB_API(cuStreamCreate);
static NB_API(cuStreamSynchronize);
static NB_API(cuStreamDestroy_v2);
#undef NB_API

static void nb_init(void) {
#define NB_LOAD(name) nb_##name = dlsym(shared_lib, #name); if (!nb_##name) { fprintf(stderr, "Neutrino missing %s\n", #name); exit(1); }
    NB_LOAD(cuLibraryLoadFromFile); NB_LOAD(cuLibraryEnumerateKernels); NB_LOAD(cuLibraryUnload);
    NB_LOAD(cuKernelGetName); NB_LOAD(cuFuncGetName); NB_LOAD(cuCtxGetCurrent); NB_LOAD(cuCtxGetDevice);
    NB_LOAD(cuCtxPushCurrent_v2); NB_LOAD(cuCtxPopCurrent_v2); NB_LOAD(cuDeviceGetUuid);
    NB_LOAD(cuDeviceGetAttribute); NB_LOAD(cuEventCreate); NB_LOAD(cuEventRecord);
    NB_LOAD(cuEventQuery); NB_LOAD(cuEventSynchronize); NB_LOAD(cuEventDestroy_v2);
    NB_LOAD(cuMemHostAlloc); NB_LOAD(cuMemFreeHost); NB_LOAD(cuMemsetD8Async);
    NB_LOAD(cuMemcpyDtoHAsync_v2); NB_LOAD(cuFuncGetAttribute); NB_LOAD(cuFuncSetAttribute);
    NB_LOAD(cuOccupancyMaxActiveBlocksPerMultiprocessor); NB_LOAD(cuStreamIsCapturing);
    NB_LOAD(cuModuleUnload);
    NB_LOAD(cuStreamCreate); NB_LOAD(cuStreamSynchronize); NB_LOAD(cuStreamDestroy_v2);
#undef NB_LOAD
}

typedef struct nb_library {
    CUlibrary handle;
    char *path;
    struct nb_library *next;
} nb_library;
typedef struct nb_kernel {
    CUkernel handle;
    nb_library *library;
    char *name;
    struct nb_kernel *next;
} nb_kernel;
typedef struct nb_function {
    CUfunction original, probed;
    CUmodule module;
    CUcontext context;
    nb_kernel *kernel;
    char workdir[4096];
    unsigned params, slots, record_bytes, record_group;
    int prepared;
    struct nb_buffer *scratch;
    struct nb_function *next;
} nb_function;
typedef struct nb_buffer {
    CUcontext context;
    CUdeviceptr device;
    void *host;
    CUevent done;
    size_t capacity, bytes;
    int pending, kernel_only;
    CUstream stream;
    uint64_t launch, executions;
    struct nb_buffer *next;
} nb_buffer;
#define NB_CLOCK_SAMPLES 32
#define NB_CLOCK_WARMUPS 4
typedef struct nb_clock_sample {
    uint64_t before, gpu, after;
} nb_clock_sample;
typedef struct nb_clock {
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUstream stream;
    CUdeviceptr device;
    uint64_t *host;
    uint64_t id, start_wall, end_wall;
    CUuuid uuid;
    int rank, ordinal;
    nb_clock_sample start[NB_CLOCK_SAMPLES], end[NB_CLOCK_SAMPLES];
    struct nb_clock *next;
} nb_clock;
static nb_clock *nb_clocks;
static uint64_t nb_clock_id;
static void nb_json_string(FILE *out, const char *s);
static nb_library *nb_libraries;
static nb_kernel *nb_kernels;
static nb_function *nb_functions;
static nb_buffer *nb_buffers;
static uint64_t nb_launch_id, nb_artifact_id;
static size_t nb_allocated;
static unsigned nb_buffer_count;
#define NB_MAX_BUFFERS 64
static size_t nb_max_bytes(void) {
    if (!nb_sync_mode()) return 64ul * 1024 * 1024;
    const char *s = getenv("NEUTRINO_SYNC_BUFFER_MB");
    unsigned long mb = s ? strtoul(s, NULL, 10) : 1024;
    if (!mb || mb > 8192) return 0;
    return mb * 1024ul * 1024;
}

static void nb_fail(const char *what) {
    nb_failed = 1; /* Error paths can hold nb_mutex; never reenter cleanup. */
    fprintf(stderr, "[neutrino barrier] %s (selected kernel NOT replayed)\n", what);
    if (event_log) { fprintf(event_log, "[barrier-error] %s\n", what); fflush(event_log); }
    exit(EXIT_FAILURE);
}
static void nb_check(CUresult r, const char *what) { if (r != CUDA_SUCCESS) { fprintf(stderr, "CUDA status %d: ", r); nb_fail(what); } }
static void *nb_calloc(size_t size) { void *p = calloc(1, size); if (!p) nb_fail("host allocation failed"); return p; }
static char *nb_string(const char *s) { char *p = strdup(s); if (!p) nb_fail("name allocation failed"); return p; }
/* Controlled kernel-duration experiment: no ordinary trace collection/replay.
 * One retained scratch buffer per function, restricted to a single stream. */
static int nb_kernel_only(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *s = getenv("NEUTRINO_BARRIER_KERNEL_ONLY");
        mode = !s ? 0 : !strcmp(s, "baseline") ? 1 : !strcmp(s, "probed") ? 2 : -1;
        if (mode < 0) nb_fail("invalid kernel-only experiment mode");
    }
    return mode;
}
static int nb_clock_mode(void) {
    const char *mode = getenv("NEUTRINO_BARRIER_CLOCK_ALIGN");
    if (!mode || !strcmp(mode, "none")) return 0;
    if (strcmp(mode, "host") || nb_kernel_only()) nb_fail("invalid clock alignment mode");
    return 1;
}
static uint64_t nb_host_time(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) nb_fail("host clock read failed");
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static nb_clock *nb_find_clock(CUcontext context) {
    for (nb_clock *c = nb_clocks; c; c = c->next) if (c->context == context) return c;
    return NULL;
}
static void nb_clock_sample_batch(nb_clock *c, nb_clock_sample *samples, int warmups) {
    void *args[] = {&c->device};
    for (int i = -warmups; i < NB_CLOCK_SAMPLES; i++) {
        uint64_t before = nb_host_time();
        nb_check(real_cuLaunchKernel(c->function, 1, 1, 1, 1, 1, 1, 0, c->stream, args, NULL), "clock sample launch");
        nb_check(nb_cuMemcpyDtoHAsync_v2(c->host, c->device, sizeof(uint64_t), c->stream), "clock sample readback");
        nb_check(nb_cuStreamSynchronize(c->stream), "clock sample completion");
        uint64_t after = nb_host_time();
        if (i >= 0) samples[i] = (nb_clock_sample){before, *c->host, after};
    }
}
static void nb_clock_write(nb_clock *c, int complete) {
    char path[4096], hostname[256] = {0}, boot[80] = {0};
    const char *session = getenv("NEUTRINO_TRACE_SESSION");
    if (!session || !*session) nb_fail("missing clock collection session");
    if (gethostname(hostname, sizeof(hostname) - 1)) nb_fail("host identity unavailable");
    FILE *in = fopen("/proc/sys/kernel/random/boot_id", "r");
    if (!in || fscanf(in, "%79s", boot) != 1) nb_fail("host boot identity unavailable");
    fclose(in);
    struct timespec resolution;
    if (clock_getres(CLOCK_MONOTONIC_RAW, &resolution)) nb_fail("host clock resolution unavailable");
    snprintf(path, sizeof(path), "%s/clock-%" PRIu64 ".json", RESULT_DIR, c->id);
    FILE *out = fopen(path, "w");
    if (!out) nb_fail("clock metadata write");
    fprintf(out, "{\"schema\":1,\"calibration_id\":%" PRIu64 ",\"context_generation\":%" PRIu64 ",\"pid\":%d,\"rank\":%d,\"device\":%d,\"device_uuid\":\"", c->id, c->id, getpid(), c->rank, c->ordinal);
    for (int i = 0; i < 16; i++) fprintf(out, "%02x", (unsigned char)c->uuid.bytes[i]);
    fprintf(out, "\",\"session_id\":"); nb_json_string(out, session);
    fprintf(out, ",\"hostname\":"); nb_json_string(out, hostname);
    fprintf(out, ",\"boot_id\":"); nb_json_string(out, boot);
    fprintf(out, ",\"clock\":\"CLOCK_MONOTONIC_RAW\",\"clock_resolution_ns\":%" PRIu64 ",\"complete\":%s,\"start_wall_ns\":%" PRIu64 ",\"end_wall_ns\":%" PRIu64,
        (uint64_t)resolution.tv_sec * 1000000000ull + (uint64_t)resolution.tv_nsec,
        complete ? "true" : "false", c->start_wall, c->end_wall);
    for (int phase = 0; phase < 2; phase++) {
        fprintf(out, ",\"%s\":[", phase ? "end" : "start");
        nb_clock_sample *samples = phase ? c->end : c->start;
        for (int i = 0; i < (phase && !complete ? 0 : NB_CLOCK_SAMPLES); i++) {
            if (i) fputc(',', out);
            fprintf(out, "{\"host_before_ns\":%" PRIu64 ",\"gpu_ns\":%" PRIu64 ",\"host_after_ns\":%" PRIu64 "}", samples[i].before, samples[i].gpu, samples[i].after);
        }
        fputc(']', out);
    }
    fprintf(out, "}\n");
    if (fclose(out)) nb_fail("clock metadata close");
}
static void nb_clock_start(CUcontext context) {
    if (!nb_clock_mode() || nb_find_clock(context)) return;
    uint64_t started = nb_host_time();
    nb_clock *c = nb_calloc(sizeof(*c));
    c->context = context; c->id = ++nb_clock_id;
    nb_check(nb_cuCtxGetDevice(&c->ordinal), "clock device");
    nb_check(nb_cuDeviceGetUuid(&c->uuid, c->ordinal), "clock device UUID");
    const char *rank = getenv("NEUTRINO_RANK");
    c->rank = rank ? atoi(rank) : c->ordinal;
    const char *ptx = ".version 6.0\n.target sm_50\n.address_size 64\n"
        ".visible .entry __nb_clock(.param .u64 output) {\n"
        ".reg .b64 p, t;\nld.param.u64 p, [output];\n"
        "mov.u64 t, %globaltimer;\nst.global.u64 [p], t;\nret;\n}\n";
    nb_check(real_cuModuleLoadData(&c->module, ptx), "load clock module");
    nb_check(real_cuModuleGetFunction(&c->function, c->module, "__nb_clock"), "clock function");
    nb_check(nb_cuStreamCreate(&c->stream, CU_STREAM_NON_BLOCKING), "clock stream");
    nb_check(real_cuMemAlloc_v2(&c->device, sizeof(uint64_t)), "clock device buffer");
    nb_check(nb_cuMemHostAlloc((void **)&c->host, sizeof(uint64_t), 0), "clock host buffer");
    nb_clock_sample_batch(c, c->start, NB_CLOCK_WARMUPS);
    c->start_wall = nb_host_time() - started;
    c->next = nb_clocks; nb_clocks = c;
    nb_clock_write(c, 0);
}
static void nb_clock_finish(CUcontext context) {
    nb_clock **cp = &nb_clocks;
    while (*cp && (*cp)->context != context) cp = &(*cp)->next;
    nb_clock *c = *cp;
    if (!c) return;
    nb_check(real_cuCtxSynchronize(), "finish trace before clock calibration");
    uint64_t started = nb_host_time();
    nb_clock_sample_batch(c, c->end, 0);
    c->end_wall = nb_host_time() - started;
    nb_clock_write(c, 1);
    nb_check(nb_cuStreamDestroy_v2(c->stream), "release clock stream");
    nb_check(real_cuMemFree_v2(c->device), "release clock device buffer");
    nb_check(nb_cuMemFreeHost(c->host), "release clock host buffer");
    nb_check(nb_cuModuleUnload(c->module), "release clock module");
    *cp = c->next; free(c);
}
static int nb_selected(const char *name) {
    if (nb_sync_mode()) return 1;
    return strstr(name, "_ZN7deep_ep2ep13dispatch_implILb") ||
           strstr(name, "_ZN7deep_ep2ep20hybrid_dispatch_implILb") ||
           strstr(name, "_ZN7deep_ep2ep12combine_implILb") ||
           strstr(name, "_ZN7deep_ep4comm12barrier_implILb");
}
static int nb_tracked(CUfunction function) {
    CUcontext context = NULL;
    nb_check(nb_cuCtxGetCurrent(&context), "tracked function context");
    pthread_mutex_lock(&nb_mutex);
    int found = 0;
    for (nb_function *f = nb_functions; f; f = f->next)
        if (f->original == function && f->context == context) { found = 1; break; }
    pthread_mutex_unlock(&nb_mutex);
    return found;
}
static nb_library *nb_find_library(CUlibrary h) {
    for (nb_library *l = nb_libraries; l; l = l->next) if (l->handle == h) return l;
    return NULL;
}
static void nb_add_kernel(CUkernel handle, nb_library *lib) {
    if (!lib) return;
    for (nb_kernel *k = nb_kernels; k; k = k->next) if (k->handle == handle && k->library == lib) return;
    const char *name = NULL;
    nb_check(nb_cuKernelGetName(&name, handle), "kernel name query");
    nb_kernel *k = nb_calloc(sizeof(*k));
    k->handle = handle; k->library = lib; k->name = nb_string(name);
    k->next = nb_kernels; nb_kernels = k;
}
static void nb_associate(CUkernel handle, CUfunction func) {
    CUcontext context;
    nb_check(nb_cuCtxGetCurrent(&context), "function context");
    pthread_mutex_lock(&nb_mutex);
    for (nb_function *f = nb_functions; f; f = f->next)
        if (f->original == func && f->context == context) { pthread_mutex_unlock(&nb_mutex); return; }
    for (nb_kernel *k = nb_kernels; k; k = k->next) {
        if (k->handle != handle || !nb_selected(k->name)) continue;
        nb_function *f = nb_calloc(sizeof(*f));
        f->original = func; f->context = context; f->kernel = k;
        f->next = nb_functions; nb_functions = f;
        break;
    }
    pthread_mutex_unlock(&nb_mutex);
}

/* Event readiness includes the async copy into retained pinned host memory.
 * Querying never waits on an unfinished launch. Lifecycle drains may wait. */
static void nb_drain_locked(CUcontext context, int wait) {
    for (nb_buffer *b = nb_buffers; b; b = b->next) {
        if (b->context != context || !b->pending) continue;
        if (b->kernel_only) {
            if (!wait) continue;
            /* Final audit only, after every measured region has completed. */
            nb_check(real_cuCtxSynchronize(), "finish kernel-only experiment");
            nb_check(real_cuMemcpyDtoH_v2(b->host, b->device, b->bytes), "audit kernel-only scratch");
            char path[4096];
            snprintf(path, sizeof(path), "%s/kernel-only-%" PRIu64 ".bin", RESULT_DIR, b->launch);
            FILE *out = fopen(path, "wb");
            if (!out || fwrite(b->host, 1, b->bytes, out) != b->bytes || fclose(out)) nb_fail("scratch audit write");
            snprintf(path, sizeof(path), "%s/kernel-only-%" PRIu64 ".calls", RESULT_DIR, b->launch);
            out = fopen(path, "w");
            if (!out) nb_fail("scratch count write");
            fprintf(out, "%" PRIu64 "\n", b->executions);
            if (fclose(out)) nb_fail("scratch count close");
            b->pending = 0;
            continue;
        }
        CUresult r = wait ? nb_cuEventSynchronize(b->done) : nb_cuEventQuery(b->done);
        if (r == CUDA_ERROR_NOT_READY) continue;
        nb_check(r, "trace completion event");
        char path[4096];
        snprintf(path, sizeof(path), "%s/barrier-%" PRIu64 ".bin", RESULT_DIR, b->launch);
        FILE *out = fopen(path, "wb");
        if (!out || fwrite(b->host, 1, b->bytes, out) != b->bytes) nb_fail("raw trace write failed");
        if (fclose(out)) nb_fail("raw trace close failed");
        b->pending = 0;
    }
}
static void nb_drain_current(void) {
    if (!nb_mode() || !nb_buffers) return;
    CUcontext context = NULL;
    if (nb_cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) return;
    pthread_mutex_lock(&nb_mutex);
    nb_drain_locked(context, 0);
    pthread_mutex_unlock(&nb_mutex);
}
static void nb_cleanup_context(CUcontext context) {
    if (!nb_mode() || !context) return;
    pthread_mutex_lock(&nb_mutex);
    nb_check(nb_cuCtxPushCurrent_v2(context), "push cleanup context");
    nb_drain_locked(context, 1);
    nb_clock_finish(context);
    nb_buffer **bp = &nb_buffers;
    while (*bp) {
        nb_buffer *b = *bp;
        if (b->context != context) { bp = &b->next; continue; }
        nb_check(real_cuMemFree_v2(b->device), "release trace buffer");
        nb_check(nb_cuMemFreeHost(b->host), "release trace host buffer");
        nb_check(nb_cuEventDestroy_v2(b->done), "release completion event");
        nb_allocated -= b->capacity; nb_buffer_count--; *bp = b->next; free(b);
    }
    nb_function **fp = &nb_functions;
    while (*fp) {
        nb_function *f = *fp;
        if (f->context != context) { fp = &f->next; continue; }
        if (f->module) nb_check(nb_cuModuleUnload(f->module), "release probe module");
        *fp = f->next; free(f);
    }
    CUcontext old; nb_check(nb_cuCtxPopCurrent_v2(&old), "pop cleanup context");
    pthread_mutex_unlock(&nb_mutex);
}
static void nb_cleanup_device(CUdevice device) {
    if (!nb_mode()) return;
    /* Only retained contexts belonging to this device are touched. */
    nb_function *f = nb_functions;
    while (f) {
        CUcontext context = f->context, old; CUdevice d;
        nb_check(nb_cuCtxPushCurrent_v2(context), "push device cleanup context");
        nb_check(nb_cuCtxGetDevice(&d), "cleanup device");
        nb_check(nb_cuCtxPopCurrent_v2(&old), "pop device cleanup context");
        if (d == device) { nb_cleanup_context(context); f = nb_functions; }
        else f = f->next;
    }
    /* Library unload may have removed every function while retaining buffers. */
    nb_buffer *b = nb_buffers;
    while (b) {
        CUcontext context = b->context, old; CUdevice d;
        nb_check(nb_cuCtxPushCurrent_v2(context), "push buffer cleanup context");
        nb_check(nb_cuCtxGetDevice(&d), "buffer cleanup device");
        nb_check(nb_cuCtxPopCurrent_v2(&old), "pop buffer cleanup context");
        if (d == device) { nb_cleanup_context(context); b = nb_buffers; }
        else b = b->next;
    }
    nb_clock *c = nb_clocks;
    while (c) {
        CUcontext context = c->context;
        if (c->ordinal == device) { nb_cleanup_context(context); c = nb_clocks; }
        else c = c->next;
    }
}

static void nb_prepare(nb_function *f) {
    if (f->prepared) return;
    snprintf(f->workdir, sizeof(f->workdir), "%s/barrier-%" PRIu64, KERNEL_DIR, ++nb_artifact_id);
    if (mkdir(f->workdir, 0755)) nb_fail("artifact directory creation");
    /* posix_spawn avoids running Python or allocator code in a forked CUDA process. */
    size_t n = 0; while (environ[n]) n++;
    char **env = nb_calloc((n + 1) * sizeof(char *));
    for (size_t i = 0, j = 0; i < n; i++)
        if (strncmp(environ[i], "LD_PRELOAD=", 11)) env[j++] = environ[i];
    char *argv[] = {NEUTRINO_PYTHON, NEUTRINO_PROBING_PY, f->workdir, f->kernel->library->path, NULL};
    pid_t child; int status;
    if (posix_spawn(&child, argv[0], NULL, NULL, argv, env)) nb_fail("probe compiler spawn");
    free(env);
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) nb_fail("PTX boundary resolution/assembly failed");
    char path[4096], name[2048];
    snprintf(path, sizeof(path), "%s/barrier.info", f->workdir);
    FILE *in = fopen(path, "r");
    if (!in || fscanf(in, "%u %u %u\n%2047s", &f->params, &f->slots, &f->record_bytes, name) != 4) nb_fail("probe metadata invalid");
    f->record_group = 1;
    int group_fields = fscanf(in, "%u", &f->record_group);
    if ((group_fields != 1 && (nb_sync_mode() || group_fields != EOF)) ||
            (f->record_group != 1 && f->record_group != 32 && (f->record_group != 0 || nb_sync_mode())))
            nb_fail("invalid synchronization recording group");
    fclose(in);
    if (strcmp(name, f->kernel->name) || f->params > 64 || f->record_bytes != 24 ||
        (nb_sync_mode() ? f->slots > 65537 : (f->slots > 4 || f->slots < 2))) nb_fail("probe ABI mismatch");
    f->prepared = 1;
    if (!f->slots) {
        fprintf(event_log, "[ptx-sync-no-sites] %s\n", name);
        fflush(event_log);
        return;
    }
    snprintf(path, sizeof(path), "%s/probed.cubin", f->workdir);
    nb_check(real_cuModuleLoad(&f->module, path), "load probed module");
    nb_check(real_cuModuleGetFunction(&f->probed, f->module, name), "load probed function");
}
static nb_buffer *nb_acquire(CUcontext context, size_t bytes) {
    nb_drain_locked(context, 0);
    for (nb_buffer *b = nb_buffers; b; b = b->next)
        if (b->context == context && !b->kernel_only && !b->pending && b->capacity >= bytes) return b;
    size_t limit = nb_max_bytes();
    if (nb_buffer_count >= NB_MAX_BUFFERS || nb_allocated > limit || bytes > limit - nb_allocated) nb_fail("pending trace storage limit exceeded");
    nb_buffer *b = nb_calloc(sizeof(*b));
    b->context = context; b->capacity = bytes;
    nb_check(real_cuMemAlloc_v2(&b->device, bytes), "allocate trace buffer");
    nb_check(nb_cuMemHostAlloc(&b->host, bytes, 0), "allocate pinned trace buffer");
    nb_check(nb_cuEventCreate(&b->done, CU_EVENT_DISABLE_TIMING), "create completion event");
    b->next = nb_buffers; nb_buffers = b; nb_allocated += bytes; nb_buffer_count++;
    return b;
}
static void nb_json_string(FILE *out, const char *s) {
    fputc('"', out);
    for (; *s; s++) { unsigned char c = *s; if (c == '"' || c == '\\') fputc('\\', out); if (c < 32) fprintf(out, "\\u%04x", c); else fputc(c, out); }
    fputc('"', out);
}
static void nb_metadata(nb_function *f, nb_buffer *b, const CUlaunchConfig *c, int registers, int active) {
    char path[4096]; CUdevice device; CUuuid uuid;
    nb_check(nb_cuCtxGetDevice(&device), "trace device"); nb_check(nb_cuDeviceGetUuid(&uuid, device), "device UUID");
    snprintf(path, sizeof(path), "%s/%s-%" PRIu64 ".json", RESULT_DIR, b->kernel_only ? "kernel-only" : "barrier", b->launch);
    FILE *out = fopen(path, "w"); if (!out) nb_fail("launch metadata write");
    const char *rank_env = getenv("NEUTRINO_RANK");
    fprintf(out, "{\"schema\":1,\"pid\":%d,\"rank\":%d,\"rank_source\":\"%s\",\"device\":%d,\"device_uuid\":\"", getpid(), rank_env ? atoi(rank_env) : device, rank_env ? "explicit" : "single-node device ordinal", device);
    for (int i = 0; i < 16; i++) fprintf(out, "%02x", (unsigned char)uuid.bytes[i]);
    fprintf(out, "\",\"launch_id\":%" PRIu64 ",\"stream\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_mem\":%u,\"registers\":%d,\"active_blocks_per_sm\":%d,\"kernel\":", b->launch, (void *)c->hStream, c->gridDimX, c->gridDimY, c->gridDimZ, c->blockDimX, c->blockDimY, c->blockDimZ, c->sharedMemBytes, registers, active);
    nb_json_string(out, f->kernel->name); fprintf(out, ",\"manifest\":");
    snprintf(path, sizeof(path), "%s/sites.json", f->workdir); nb_json_string(out, path);
    fprintf(out, ",\"attributes\":[");
    for (unsigned i = 0; i < c->numAttrs; i++) {
        if (i) fputc(',', out);
        fprintf(out, "{\"id\":%d,\"raw\":\"", c->attrs[i].id);
        for (size_t j = 0; j < sizeof(c->attrs[i].value); j++) fprintf(out, "%02x", ((unsigned char *)&c->attrs[i].value)[j]);
        fprintf(out, "\"}");
    }
    fprintf(out, "],\"kernel_only_mode\":%d", b->kernel_only);
    nb_clock *clock = nb_find_clock(f->context);
    if (clock) {
        fprintf(out, ",\"clock_alignment\":\"host\",\"calibration_id\":%" PRIu64 ",\"context_generation\":%" PRIu64 ",\"session_id\":", clock->id, clock->id);
        nb_json_string(out, getenv("NEUTRINO_TRACE_SESSION"));
    } else fprintf(out, ",\"clock_alignment\":\"none\"");
    fprintf(out, "}\n"); if (fclose(out)) nb_fail("launch metadata close");
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *path, CUjit_option *opts, void **values, unsigned nopts, CUlibraryOption *lopts, void **lvalues, unsigned nlopts) {
    if (!shared_lib) init();
    CUresult r = nb_cuLibraryLoadFromFile(library, path, opts, values, nopts, lopts, lvalues, nlopts);
    if (r == CUDA_SUCCESS && nb_mode()) {
        nb_library *l = nb_calloc(sizeof(*l)); l->handle = *library;
        l->path = realpath(path, NULL); if (!l->path) nb_fail("loaded artifact path unavailable");
        pthread_mutex_lock(&nb_mutex); l->next = nb_libraries; nb_libraries = l; pthread_mutex_unlock(&nb_mutex);
    }
    return r;
}
CUresult cuLibraryEnumerateKernels(CUkernel *kernels, unsigned count, CUlibrary library) {
    if (!shared_lib) init();
    CUresult r = nb_cuLibraryEnumerateKernels(kernels, count, library);
    if (r == CUDA_SUCCESS && nb_mode()) {
        pthread_mutex_lock(&nb_mutex);
        for (unsigned i = 0; i < count; i++) nb_add_kernel(kernels[i], nb_find_library(library));
        pthread_mutex_unlock(&nb_mutex);
    }
    return r;
}
CUresult cuLibraryUnload(CUlibrary library) {
    if (!shared_lib) init();
    if (nb_mode()) {
        pthread_mutex_lock(&nb_mutex);
        nb_function **fp = &nb_functions;
        while (*fp) {
            nb_function *f = *fp;
            if (f->kernel->library->handle != library) { fp = &f->next; continue; }
            CUcontext old; nb_check(nb_cuCtxPushCurrent_v2(f->context), "push library context");
            nb_drain_locked(f->context, 1);
            if (f->module) nb_check(nb_cuModuleUnload(f->module), "unload probe module");
            nb_check(nb_cuCtxPopCurrent_v2(&old), "pop library context"); *fp = f->next; free(f);
        }
        nb_kernel **kp = &nb_kernels;
        while (*kp) { nb_kernel *k = *kp; if (k->library->handle == library) { *kp = k->next; free(k->name); free(k); } else kp = &k->next; }
        nb_library **lp = &nb_libraries;
        while (*lp) { nb_library *l = *lp; if (l->handle == library) { *lp = l->next; free(l->path); free(l); } else lp = &l->next; }
        pthread_mutex_unlock(&nb_mutex);
    }
    return nb_cuLibraryUnload(library);
}
CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction function, void **params, void **extra) {
    if (!shared_lib) init();
    if (!nb_mode()) return real_cuLaunchKernelEx(config, function, params, extra);
    CUcontext context; nb_check(nb_cuCtxGetCurrent(&context), "launch context");
    pthread_mutex_lock(&nb_mutex);
    nb_function *f;
    for (f = nb_functions; f; f = f->next) if (f->original == function && f->context == context) break;
    if (!f) {
        const char *name = NULL;
        nb_check(nb_cuFuncGetName(&name, function), "coverage function name");
        if (!nb_sync_mode() && nb_selected(name)) nb_fail("selected function has no exact artifact association");
        if (nb_sync_mode()) {
            static int warned;
            if (!warned++) { fprintf(event_log, "[ptx-sync-coverage] In-memory and module-loaded kernels are outside file-library coverage\n"); fflush(event_log); }
        }
        pthread_mutex_unlock(&nb_mutex);
        return real_cuLaunchKernelEx(config, function, params, extra);
    }
    if (extra || (!params && !nb_sync_mode()) || NEUTRINO_BENCHMARK || DYNAMIC) nb_fail("unsupported launch ABI/replay mode");
    CUstreamCaptureStatus capture;
    nb_check(nb_cuStreamIsCapturing(config->hStream, &capture), "capture query");
    if (capture != CU_STREAM_CAPTURE_STATUS_NONE) nb_fail("CUDA Graph capture is out of scope");
    if (!nb_exit_registered) {
        /* CUDA registers its own shutdown during initialization. Register later
         * so pending traces drain before the real driver becomes unavailable. */
        if (atexit(nb_shutdown)) nb_fail("shutdown registration failed");
        nb_exit_registered = 1;
    }
    nb_prepare(f);
    if (f->params && !params) nb_fail("missing original kernel arguments");
    if (!f->slots) {
        pthread_mutex_unlock(&nb_mutex);
        return real_cuLaunchKernelEx(config, function, params, extra);
    }
    int dynamic_smem, carveout, registers, active, sms; CUdevice device;
    nb_check(nb_cuFuncGetAttribute(&dynamic_smem, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function), "shared memory attribute");
    nb_check(nb_cuFuncSetAttribute(f->probed, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_smem), "copy shared memory attribute");
    nb_check(nb_cuFuncGetAttribute(&carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, function), "carveout attribute");
    nb_check(nb_cuFuncSetAttribute(f->probed, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, carveout), "copy carveout attribute");
    int experiment = nb_kernel_only();
    CUfunction launched = experiment == 1 ? function : f->probed;
    nb_check(nb_cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, launched), "register count");
    uint64_t blocks = (uint64_t)config->gridDimX * config->gridDimY * config->gridDimZ;
    uint64_t threads = (uint64_t)config->blockDimX * config->blockDimY * config->blockDimZ;
    if (!blocks || blocks > 65535 || !threads || threads > 1024) nb_fail("unsupported launch dimensions");
    nb_check(nb_cuOccupancyMaxActiveBlocksPerMultiprocessor(&active, launched, threads, config->sharedMemBytes), "occupancy query");
    nb_check(nb_cuCtxGetDevice(&device), "launch device");
    nb_check(nb_cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "SM count");
    int cooperative = 0;
    for (unsigned i = 0; i < config->numAttrs; i++)
        if (config->attrs[i].id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE) cooperative = config->attrs[i].value.cooperative;
    if ((!cooperative && !nb_sync_mode()) || (cooperative && blocks > (uint64_t)active * sms)) nb_fail("cooperative residency requirement not met");
    nb_clock_start(context);
    /* Group zero denotes one compact thread-0 record per block (aware probes only). */
    uint64_t recording_units = f->record_group ? (threads + f->record_group - 1) / f->record_group : 1;
    size_t bytes = blocks * recording_units * f->slots * f->record_bytes;
    if (experiment) {
        nb_buffer *b = f->scratch;
        if (!b) {
            b = f->scratch = nb_acquire(context, bytes);
            b->kernel_only = experiment; b->stream = config->hStream;
            b->bytes = bytes; b->launch = ++nb_launch_id;
            nb_check(nb_cuMemsetD8Async(b->device, 0, bytes, config->hStream), "initialize experiment scratch");
            nb_metadata(f, b, config, registers, active);
        }
        if (b->stream != config->hStream || b->bytes != bytes || b->executions == UINT32_MAX)
            nb_fail("kernel-only experiment requires fixed dimensions and one ordered stream per function");
        void *args[65]; if (f->params) memcpy(args, params, f->params * sizeof(void *)); args[f->params] = &b->device;
        CUresult r = real_cuLaunchKernelEx(config, launched, experiment == 1 ? params : args, NULL);
        nb_check(r, "kernel-only experiment launch failed");
        b->executions++; b->pending = 1;
        pthread_mutex_unlock(&nb_mutex);
        return r;
    }
    nb_buffer *b = nb_acquire(context, bytes); b->bytes = bytes; b->launch = ++nb_launch_id;
    void *args[65]; if (f->params) memcpy(args, params, f->params * sizeof(void *)); args[f->params] = &b->device;
    nb_check(nb_cuMemsetD8Async(b->device, 0, bytes, config->hStream), "clear fixed slots");
    /* The exact config pointer preserves ALL launch attributes, including PDL. */
    CUresult r = real_cuLaunchKernelEx(config, f->probed, args, NULL);
    nb_check(r, "instrumented launch failed");
    nb_check(nb_cuMemcpyDtoHAsync_v2(b->host, b->device, bytes, config->hStream), "enqueue trace copy");
    nb_check(nb_cuEventRecord(b->done, config->hStream), "record completion event");
    b->pending = 1;
    nb_metadata(f, b, config, registers, active);
    pthread_mutex_unlock(&nb_mutex);
    return r;
}

/* Redirect only exact real API addresses, retaining versioned and PTDS ABI.
 * APIs outside this list keep the driver's own resolution semantics. */
static void nb_redirect(void **pfn) {
    if (!nb_mode() || !*pfn) return;
    const char *names[] = {"cuLibraryLoadFromFile", "cuLibraryEnumerateKernels", "cuLibraryUnload",
        "cuLibraryGetKernel", "cuKernelGetFunction", "cuLaunchKernelEx", "cuLaunchKernel",
        "cuLaunchKernelEx_ptsz", "cuLaunchKernel_ptsz",
        "cuCtxSynchronize", "cuStreamSynchronize", "cuStreamSynchronize_ptsz", "cuEventSynchronize",
        "cuCtxDestroy", "cuCtxDestroy_v2", "cuDevicePrimaryCtxRelease", "cuDevicePrimaryCtxRelease_v2",
        "cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxReset_v2", NULL};
    for (int i = 0; names[i]; i++) {
        if (*pfn != dlsym(shared_lib, names[i])) continue;
        void *self = dlopen(getenv("NEUTRINO_HOOK_DRIVER"), RTLD_LAZY | RTLD_NOLOAD);
        if (!self) nb_fail("hook address resolution");
        void *hook = dlsym(self, names[i]); dlclose(self);
        if (!hook) nb_fail("missing ABI-compatible hook");
        *pfn = hook;
        return;
    }
}
CUresult cuGetProcAddress(const char *symbol, void **pfn, int version, cuuint64_t flags) {
    if (!shared_lib) init();
    CUresult (*real)(const char *, void **, int, cuuint64_t) = dlsym(shared_lib, "cuGetProcAddress");
    CUresult r = real(symbol, pfn, version, flags);
    if (r == CUDA_SUCCESS) nb_redirect(pfn);
    return r;
}
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int version, cuuint64_t flags, CUdriverProcAddressQueryResult *status) {
    if (!shared_lib) init();
    CUresult (*real)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *) = dlsym(shared_lib, "cuGetProcAddress_v2");
    CUresult r = real(symbol, pfn, version, flags, status);
    if (r == CUDA_SUCCESS) nb_redirect(pfn);
    return r;
}

__attribute__((destructor)) static void nb_shutdown(void) {
    if (!shared_lib || !nb_mode() || nb_failed) return;
    while (nb_buffers) nb_cleanup_context(nb_buffers->context);
    while (nb_functions) nb_cleanup_context(nb_functions->context);
    while (nb_clocks) nb_cleanup_context(nb_clocks->context);
}
