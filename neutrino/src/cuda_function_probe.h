/* CUDA function-probe image ownership. Included only by cuda.c after signatures.
 * No application-specific selectors belong here. All driver calls occur outside
 * nt_mutex; retained image views remain valid across unload and handle reuse.
 */
#ifndef NEUTRINO_CUDA_FUNCTION_PROBE_H
#define NEUTRINO_CUDA_FUNCTION_PROBE_H
#include <errno.h>
#include <limits.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <spawn.h>
extern char **environ;

#define NT_IMAGE_LIMIT ((size_t)512 * 1024 * 1024)

enum nt_kind { NT_LIBRARY, NT_MODULE, NT_KERNEL, NT_FUNCTION };
typedef struct nt_image {
    unsigned char *bytes;
    size_t size, refs;
    char digest[41];
} nt_image;

typedef struct nt_binding {
    enum nt_kind kind;
    void *handle;
    CUcontext context;
    uint64_t id, parent;
    bool closing;
    nt_image *image;
    char *name;
    struct nt_binding *next;
} nt_binding;

typedef struct {
    nt_image *image;
    char *name;
    uint64_t binding_id, owner_id;
    CUcontext context;
} nt_view;

static pthread_mutex_t nt_mutex = PTHREAD_MUTEX_INITIALIZER;
static nt_binding *nt_bindings;
static uint64_t nt_binding_serial, nt_launch_serial;
static int nt_enabled;
static FILE *nt_coverage, *nt_submissions;

typedef struct { size_t size, cap; } nt_map;
typedef struct nt_artifact {
    nt_image *image;
    char *name;
    char path[PATH_MAX];
    int status; /* 0 preparing, 1 instrumented, 2 unselected, -1 unsupported */
    unsigned int nparams, nmaps;
    size_t metadata_stride;
    nt_map *maps;
    struct nt_artifact *next;
} nt_artifact;

typedef struct nt_loaded {
    nt_artifact *artifact;
    CUcontext context;
    unsigned long long context_id;
    CUdevice device;
    uint64_t binding_id, owner_id;
    CUmodule module;
    CUfunction function;
    size_t users;
    bool closing, quarantined;
    int status;
    struct nt_loaded *next;
} nt_loaded;

#define NT_MAX_MAPS 1024
#define NT_MAX_PARAMS 4096
#define NT_MAX_ATTRS 1024
#define NT_GAPS 256

typedef struct nt_pool nt_pool;
typedef struct nt_device_slab {
    CUdeviceptr pointer;
    CUevent ready;
    size_t capacity, charge;
    int state; /* 0 ready, 1 owned, 2 resetting, 3 quarantined */
    bool unproven_kernel;
    uint64_t sequence;
} nt_device_slab;
typedef struct nt_host_slab {
    void *pointer;
    size_t capacity, charge;
    int state; /* 0 ready, 1 owned, 3 quarantined */
} nt_host_slab;
enum nt_job_state { NT_FREE, NT_RESERVED, NT_KERNEL_PENDING, NT_COPY_PENDING,
                    NT_WORKING, NT_EXPORT, NT_WRITING, NT_QUARANTINED };
typedef struct nt_job {
    nt_loaded *loaded;
    nt_pool *pool;
    nt_device_slab *slab;
    nt_host_slab *staging;
    enum nt_job_state state;
    uint64_t sequence;
    CUevent copy_event;
    CUdeviceptr device_storage[NT_MAX_MAPS];
    void *host_storage[NT_MAX_MAPS], *arguments[NT_MAX_PARAMS + NT_MAX_MAPS];
    size_t size_storage[NT_MAX_MAPS];
    CUlaunchAttribute attribute_storage[NT_MAX_ATTRS];
    uint64_t id;
    CUdeviceptr *device;
    void **host;
    size_t *sizes;
    unsigned int nmaps;
    CUevent event;
    size_t bytes, payload_bytes, charge, warps;
    CUlaunchConfig config;
    CUlaunchAttribute *attributes;
    bool submitted, copied, quarantined;
    CUcontext context;
    CUresult launch_result;
    const char *failure;
    CUstream completion_stream;
    pthread_t submitter;
    struct nt_job *next, *active_next, *journal_next;
} nt_job;

static nt_artifact *nt_artifacts;
static nt_loaded *nt_loaded_images;
static nt_job *nt_active;
typedef struct nt_capacity {
    size_t bytes, count;
    bool preparing;
    nt_device_slab *device;
    nt_host_slab *host;
    nt_job *jobs;
    struct nt_capacity *next;
} nt_capacity;
struct nt_pool {
    CUcontext context;
    unsigned long long generation;
    uint64_t first_error_sequence;
    CUstream stream;
    pthread_t worker;
    bool preparing, closing, stopped, joining, worker_started;
    nt_capacity *classes;
    struct nt_pool *next;
};
static nt_pool *nt_pools;
static unsigned int nt_prewarm = 8;
static pthread_t nt_writer;
static pthread_once_t nt_writer_once = PTHREAD_ONCE_INIT;
static bool nt_writer_started, nt_writer_stop;
static nt_job *nt_exports, **nt_export_tail = &nt_exports;
static nt_job *nt_journals, **nt_journal_tail = &nt_journals;
typedef struct {
    uint64_t id;
    CUcontext context;
    CUfunction function;
    char kernel[1024];
    const char *reason;
} nt_gap_record;
static nt_gap_record nt_gaps[NT_GAPS];
static size_t nt_gap_head, nt_gap_count;
static uint64_t nt_gap_overflow;
static struct {
    uint64_t accepted, completed, skipped, hits, misses, errors;
    uint64_t device_allocations, host_allocations, event_allocations, cold_ns;
    size_t active, high_water, resident_high_water;
} nt_stats;
static struct { const char *reason; uint64_t count; } nt_skip_reasons[64];
static pthread_cond_t nt_changed = PTHREAD_COND_INITIALIZER;
static bool nt_stopping;
static size_t nt_budget = 128 * 1024 * 1024, nt_inflight;
static uint64_t nt_artifact_serial;

typedef struct nt_preference {
    uint64_t binding_id;
    CUdevice device;
    bool shared;
    int value;
    struct nt_preference *next;
} nt_preference;
static nt_preference *nt_preferences;

typedef struct nt_primary {
    CUdevice device;
    CUcontext context;
    size_t retiring;
    struct nt_primary *next;
} nt_primary;
static nt_primary *nt_primaries;

/* Caller holds nt_mutex. Records remain stable across overlapping API calls. */
static bool nt_context_closing(CUcontext context) {
    for (nt_primary *p = nt_primaries; p; p = p->next)
        if (p->context == context && p->retiring) return true;
    return false;
}

static bool nt_read_memory(void *dst, const void *src, size_t size) {
    struct iovec local = {dst, size}, remote = {(void *)src, size};
    return syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1, 0) == (ssize_t)size;
}

static bool nt_extent(size_t offset, size_t size, size_t *extent) {
    if (offset > NT_IMAGE_LIMIT || size > NT_IMAGE_LIMIT - offset) return false;
    if (offset + size > *extent) *extent = offset + size;
    return true;
}

static bool nt_elf_header(const Elf64_Ehdr *h) {
    return !memcmp(h->e_ident, "\177ELF\2\1\1", 7) && h->e_machine == 190 &&
           h->e_version == 1 && h->e_ehsize == sizeof(*h) && h->e_shnum &&
           h->e_shentsize == sizeof(Elf64_Shdr) &&
           (!h->e_phnum || h->e_phentsize == sizeof(Elf64_Phdr));
}

static nt_image *nt_image_new(unsigned char *bytes, size_t size) {
    nt_image *image = calloc(1, sizeof(*image));
    if (!image) { free(bytes); return NULL; }
    image->bytes = bytes;
    image->size = size;
    image->refs = 1;
    SHA1_CTX hash;
    BYTE digest[SHA1_BLOCK_SIZE];
    sha1_init(&hash);
    sha1_update(&hash, bytes, size);
    sha1_final(&hash, digest);
    for (size_t i = 0; i < sizeof(digest); ++i)
        snprintf(image->digest + i * 2, 3, "%02x", digest[i]);
    return image;
}

/* The data-load ABI has no length. Fault-contained reads and checked ELF extents
 * avoid dereferencing malformed headers in the application address space.
 * Fatbins/PTX remain explicit unsupported images, never guessed cubins.
 */
static nt_image *nt_snapshot_data(const void *data) {
    if (!nt_enabled || !data) return NULL;
    Elf64_Ehdr h;
    size_t extent = sizeof(h);
    if (!nt_read_memory(&h, data, sizeof(h)) || !nt_elf_header(&h) ||
        !nt_extent(h.e_shoff, (size_t)h.e_shnum * h.e_shentsize, &extent) ||
        !nt_extent(h.e_phoff, (size_t)h.e_phnum * h.e_phentsize, &extent)) return NULL;
    for (size_t i = 0; i < h.e_shnum; ++i) {
        Elf64_Shdr s;
        if (!nt_read_memory(&s, (const char *)data + h.e_shoff + i * sizeof(s), sizeof(s)) ||
            (s.sh_type != SHT_NOBITS && !nt_extent(s.sh_offset, s.sh_size, &extent))) return NULL;
    }
    for (size_t i = 0; i < h.e_phnum; ++i) {
        Elf64_Phdr p;
        if (!nt_read_memory(&p, (const char *)data + h.e_phoff + i * sizeof(p), sizeof(p)) ||
            !nt_extent(p.p_offset, p.p_filesz, &extent)) return NULL;
    }
    unsigned char *bytes = malloc(extent);
    if (!bytes) return NULL;
    if (!nt_read_memory(bytes, data, extent)) { free(bytes); return NULL; }
    return nt_image_new(bytes, extent);
}

static nt_image *nt_snapshot_file(const char *path) {
    if (!nt_enabled || !path) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    struct stat st;
    unsigned char *bytes = NULL;
    nt_image *image = NULL;
    if (fstat(fileno(file), &st) || st.st_size < (off_t)sizeof(Elf64_Ehdr) ||
        st.st_size > NT_IMAGE_LIMIT || !S_ISREG(st.st_mode)) goto done;
    bytes = malloc(st.st_size);
    if (!bytes || fread(bytes, 1, st.st_size, file) != (size_t)st.st_size ||
        !nt_elf_header((const Elf64_Ehdr *)bytes)) goto done;
    image = nt_image_new(bytes, st.st_size);
    bytes = NULL;
done:
    free(bytes);
    fclose(file);
    return image;
}

/* Caller holds nt_mutex, except for unpublished snapshots. */
static void nt_image_release(nt_image *image) {
    if (image && --image->refs == 0) { free(image->bytes); free(image); }
}

static nt_binding *nt_find(enum nt_kind kind, void *handle, CUcontext context) {
    for (nt_binding *b = nt_bindings; b; b = b->next)
        if (b->kind == kind && b->handle == handle && b->context == context) return b;
    return NULL;
}

/* Children use generation IDs, not raw handles. Retiring a reused handle cannot
 * remove associations belonging to a later generation or another CUDA context.
 */
static void nt_remove(uint64_t id) {
    for (;;) {
        nt_binding *child = nt_bindings;
        while (child && child->parent != id) child = child->next;
        if (!child) break;
        nt_remove(child->id);
    }
    nt_binding **link = &nt_bindings;
    while (*link && (*link)->id != id) link = &(*link)->next;
    if (*link) {
        nt_binding *b = *link;
        *link = b->next;
        nt_preference **pref = &nt_preferences;
        while (*pref) {
            nt_preference *p = *pref;
            if (p->binding_id != b->id) { pref = &p->next; continue; }
            *pref = p->next; free(p);
        }
        nt_image_release(b->image);
        free(b->name);
        free(b);
    }
}

static CUcontext nt_current(void) {
    CUcontext context = NULL;
    if (real_cuCtxGetCurrent(&context) != CUDA_SUCCESS) return NULL;
    return context;
}

/* Takes ownership of a snapshot even when the CUDA call failed. */
static void nt_register(enum nt_kind kind, void *handle, CUcontext context,
                        nt_image *image, CUresult result) {
    if (!nt_enabled) { nt_image_release(image); return; }
    pthread_mutex_lock(&nt_mutex);
    if (result == CUDA_SUCCESS) {
        nt_binding *old = nt_find(kind, handle, context);
        if (old && old->image && image && old->image->size == image->size &&
            !strcmp(old->image->digest, image->digest) &&
            !memcmp(old->image->bytes, image->bytes, image->size)) {
            nt_image_release(image);
            pthread_mutex_unlock(&nt_mutex);
            return;
        }
        if (old) nt_remove(old->id);
        nt_binding *b = calloc(1, sizeof(*b));
        if (b) {
            *b = (nt_binding){.kind=kind, .handle=handle, .context=context,
                .id=++nt_binding_serial, .image=image, .next=nt_bindings};
            nt_bindings = b;
            image = NULL;
        }
    }
    nt_image_release(image);
    pthread_mutex_unlock(&nt_mutex);
}

static void nt_bind(enum nt_kind parent_kind, void *parent_handle, CUcontext parent_context,
                    enum nt_kind kind, void *handle, CUcontext context, const char *name) {
    if (!nt_enabled) return;
    pthread_mutex_lock(&nt_mutex);
    nt_binding *parent = nt_find(parent_kind, parent_handle, parent_context);
    if (parent) {
        nt_binding *old = nt_find(kind, handle, context);
        const char *effective_name = name ? name : parent->name;
        /* Repeated lookup must not erase descendant mappings or their caches. */
        if (old && old->parent == parent->id &&
            ((!effective_name && !old->name) || (effective_name && old->name &&
             !strcmp(effective_name, old->name)))) goto done;
        char *copy = effective_name ? strdup(effective_name) : NULL;
        if ((name || parent->name) && !copy) goto done;
        nt_binding *b = calloc(1, sizeof(*b));
        if (!b) { free(copy); goto done; }
        if (old) nt_remove(old->id);
        *b = (nt_binding){.kind=kind, .handle=handle, .context=context,
            .id=++nt_binding_serial, .parent=parent->id, .image=parent->image,
            .name=copy, .next=nt_bindings};
        if (b->image) ++b->image->refs;
        nt_bindings = b;
    }
done:
    pthread_mutex_unlock(&nt_mutex);
}

static void nt_forget(enum nt_kind kind, void *handle, CUcontext context) {
    if (!nt_enabled) return;
    pthread_mutex_lock(&nt_mutex);
    nt_binding *b = nt_find(kind, handle, context);
    if (b) nt_remove(b->id);
    pthread_mutex_unlock(&nt_mutex);
}

static void nt_forget_context(CUcontext context) {
    if (!nt_enabled) return;
    pthread_mutex_lock(&nt_mutex);
    for (;;) {
        nt_binding *b = nt_bindings;
        while (b && b->context != context) b = b->next;
        if (!b) break;
        nt_remove(b->id);
    }
    pthread_mutex_unlock(&nt_mutex);
}

static nt_view nt_lookup(CUfunction function, CUcontext context) {
    nt_view view = {.context=context};
    pthread_mutex_lock(&nt_mutex);
    nt_binding *b = nt_find(NT_FUNCTION, function, context);
    if (b) {
        view.binding_id = b->id;
        nt_binding *root = b;
        while (root->parent) {
            nt_binding *parent = nt_bindings;
            while (parent && parent->id != root->parent) parent = parent->next;
            if (!parent) break;
            root = parent;
        }
        if (b->closing || root->closing || nt_context_closing(context)) {
            pthread_mutex_unlock(&nt_mutex);
            return view;
        }
        view.owner_id = root->id;
        view.image = b->image;
        view.name = b->name ? strdup(b->name) : NULL;
        if (view.image) ++view.image->refs;
    }
    pthread_mutex_unlock(&nt_mutex);
    return view;
}

static void nt_view_release(nt_view *view) {
    pthread_mutex_lock(&nt_mutex);
    nt_image_release(view->image);
    pthread_mutex_unlock(&nt_mutex);
    free(view->name);
    memset(view, 0, sizeof(*view));
}

static void nt_json_string(FILE *file, const char *text) {
    if (!text) { fputs("null", file); return; }
    fputc('"', file);
    for (const unsigned char *s = (const unsigned char *)text; *s; ++s) {
        if (*s == '"' || *s == '\\') { fputc('\\', file); fputc(*s, file); }
        else if (*s < 32) fprintf(file, "\\u%04x", *s);
        else fputc(*s, file);
    }
    fputc('"', file);
}

static void nt_start_writer(void);

static void nt_gap(CUfunction function, const char *reason) {
    if (!nt_enabled) return;
    pthread_once(&nt_writer_once, nt_start_writer);
    CUcontext context = nt_current();
    pthread_mutex_lock(&nt_mutex);
    ++nt_stats.skipped;
    for (size_t i = 0; i < 64; ++i) {
        if (!nt_skip_reasons[i].reason) nt_skip_reasons[i].reason = reason;
        if (!strcmp(nt_skip_reasons[i].reason, reason)) { ++nt_skip_reasons[i].count; break; }
    }
    uint64_t launch = ++nt_launch_serial;
    if (nt_gap_count < NT_GAPS) {
        nt_gap_record *g = &nt_gaps[(nt_gap_head + nt_gap_count++) % NT_GAPS];
        *g = (nt_gap_record){.id=launch, .context=context, .function=function, .reason=reason};
        nt_binding *binding = nt_find(NT_FUNCTION, function, context);
        if (binding && binding->name) snprintf(g->kernel, sizeof(g->kernel), "%s", binding->name);
    } else ++nt_gap_overflow;
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
}

static bool nt_multiply(size_t a, size_t b, size_t *out) {
    if (b && a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

static bool nt_name_matches(const char *list, const char *name) {
    if (!list || !*list || !name) return false;
    char *copy = strdup(list), *state = NULL;
    if (!copy) return false;
    bool found = false;
    for (char *part = strtok_r(copy, ":", &state); part; part = strtok_r(NULL, ":", &state))
        if (strstr(name, part)) { found = true; break; }
    free(copy);
    return found;
}

static void nt_compile(nt_artifact *artifact) {
    char path[PATH_MAX];
    if (mkdir(artifact->path, 0755)) return;
    if (snprintf(path, sizeof(path), "%s/original.bin", artifact->path) >= sizeof(path)) return;
    FILE *file = fopen(path, "wb");
    if (!file) return;
    bool wrote = fwrite(artifact->image->bytes, 1, artifact->image->size, file) == artifact->image->size;
    if (fclose(file) || !wrote) return;
    char *argv[] = {NEUTRINO_PYTHON, "-m", "neutrino.probe.cuda_function", artifact->path, artifact->name, NULL};
    pid_t pid;
    if (posix_spawnp(&pid, NEUTRINO_PYTHON, NULL, NULL, argv, environ)) return;
    int status;
    while (waitpid(pid, &status, 0) < 0) if (errno != EINTR) return;
    if (!WIFEXITED(status)) return;
    if (WEXITSTATUS(status) == 2) { artifact->status = 2; return; }
    if (WEXITSTATUS(status)) return;
    if (snprintf(path, sizeof(path), "%s/function.info", artifact->path) >= sizeof(path)) return;
    file = fopen(path, "r");
    if (!file) return;
    unsigned int version;
    bool valid = fscanf(file, "%u%u%u%zu", &version, &artifact->nparams, &artifact->nmaps,
                        &artifact->metadata_stride) == 4 && version == 3 &&
                 artifact->nparams <= 4096 && artifact->nmaps > 0 && artifact->nmaps <= 1024 &&
                 artifact->metadata_stride == 0;
    if (valid) artifact->maps = calloc(artifact->nmaps, sizeof(nt_map));
    valid = valid && artifact->maps;
    for (unsigned int i = 0; valid && i < artifact->nmaps; ++i) {
        nt_map *map = &artifact->maps[i];
        size_t bytes;
        valid = fscanf(file, "%zu,%zu", &map->size, &map->cap) == 2 &&
                map->size && map->cap && !(map->size % 8) &&
                nt_multiply(map->size, map->cap, &bytes) && bytes <= UINT32_MAX;
    }
    fclose(file);
    if (valid) artifact->status = 1;
}

/* One immutable probe configuration/tool version per process. Full cubin bytes
 * (which include the declared target) plus the exact kernel key this cache;
 * function.json records the compiler's version/options/assembler SHA-256 key.
 */
static nt_artifact *nt_artifact_get(const nt_view *view) {
    pthread_mutex_lock(&nt_mutex);
    nt_artifact *a = nt_artifacts;
    for (; a; a = a->next) {
        if (!strcmp(a->name, view->name) && a->image->size == view->image->size &&
            !strcmp(a->image->digest, view->image->digest) &&
            !memcmp(a->image->bytes, view->image->bytes, view->image->size)) break;
    }
    if (a) {
        while (!a->status) pthread_cond_wait(&nt_changed, &nt_mutex);
        pthread_mutex_unlock(&nt_mutex);
        return a;
    }
    a = calloc(1, sizeof(*a));
    if (!a) { pthread_mutex_unlock(&nt_mutex); return NULL; }
    a->name = strdup(view->name);
    if (!a->name || snprintf(a->path, sizeof(a->path), "%s/function-%llu-%s", KERNEL_DIR,
                            (unsigned long long)++nt_artifact_serial, view->image->digest) >= sizeof(a->path)) {
        free(a->name); free(a); pthread_mutex_unlock(&nt_mutex); return NULL;
    }
    a->image = view->image;
    ++a->image->refs;
    a->next = nt_artifacts;
    nt_artifacts = a;
    pthread_mutex_unlock(&nt_mutex);
    /* Only this preparing thread writes the private descriptor. Publish status
     * under nt_mutex after compilation, so readers never observe partial maps. */
    nt_artifact prepared = *a;
    prepared.status = -1;
    nt_compile(&prepared);
    pthread_mutex_lock(&nt_mutex);
    a->nparams = prepared.nparams;
    a->nmaps = prepared.nmaps;
    a->metadata_stride = prepared.metadata_stride;
    a->maps = prepared.maps;
    a->status = prepared.status;
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
    return a;
}

static void nt_loaded_release(nt_loaded *loaded) {
    pthread_mutex_lock(&nt_mutex);
    --loaded->users;
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
}

static nt_loaded *nt_loaded_get(const nt_view *view, nt_artifact *artifact, unsigned long long context_id) {
    pthread_mutex_lock(&nt_mutex);
    nt_binding *binding = NULL, *owner = NULL;
    for (nt_binding *b = nt_bindings; b; b = b->next) {
        if (b->id == view->binding_id) binding = b;
        if (b->id == view->owner_id) owner = b;
    }
    if (nt_stopping || !binding || !owner || binding->closing || owner->closing ||
        nt_context_closing(view->context)) {
        pthread_mutex_unlock(&nt_mutex); return NULL;
    }
    nt_loaded *loaded = nt_loaded_images;
    for (; loaded; loaded = loaded->next)
        if (loaded->binding_id == view->binding_id && loaded->context == view->context &&
            loaded->context_id == context_id) break;
    if (loaded) {
        if (loaded->closing) { pthread_mutex_unlock(&nt_mutex); return NULL; }
        ++loaded->users;
        while (!loaded->status) pthread_cond_wait(&nt_changed, &nt_mutex);
        pthread_mutex_unlock(&nt_mutex);
        return loaded;
    }
    loaded = calloc(1, sizeof(*loaded));
    if (!loaded) { pthread_mutex_unlock(&nt_mutex); return NULL; }
    *loaded = (nt_loaded){.artifact=artifact, .context=view->context, .context_id=context_id,
        .binding_id=view->binding_id, .owner_id=view->owner_id, .users=1, .next=nt_loaded_images};
    nt_loaded_images = loaded;
    pthread_mutex_unlock(&nt_mutex);
    char path[PATH_MAX];
    int status = -1;
    if (snprintf(path, sizeof(path), "%s/probed.bin", artifact->path) >= sizeof(path)) goto done;
    if (real_cuCtxGetDevice(&loaded->device) != CUDA_SUCCESS ||
        real_cuModuleLoad(&loaded->module, path) != CUDA_SUCCESS ||
        real_cuModuleGetFunction(&loaded->function, loaded->module, artifact->name) != CUDA_SUCCESS) goto done;
    status = 1;
done:
    pthread_mutex_lock(&nt_mutex);
    loaded->status = status;
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
    return loaded;
}

/* The companion is the commit record: readers must not consume a payload until
 * its JSON exists. Binary headers and map layouts remain the legacy ABI. */
static bool nt_write_job(nt_job *job) {
    nt_artifact *a = job->loaded->artifact;
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/function-%llu.bin", RESULT_DIR,
                 (unsigned long long)job->id) >= sizeof(path)) return false;
    FILE *file = fopen(path, "wb");
    if (!file) return false;
    CUlaunchConfig *c = &job->config;
    trace_header_t header = {c->gridDimX, c->gridDimY, c->gridDimZ,
        c->blockDimX, c->blockDimY, c->blockDimZ, c->sharedMemBytes, a->nmaps};
    trace_section_t *sections = calloc(a->nmaps, sizeof(*sections));
    if (!sections) { fclose(file); return false; }
    for (unsigned int i = 0; i < a->nmaps; ++i) {
        sections[i].size = a->maps[i].size * a->maps[i].cap;
        sections[i].warpDiv = 32;
    }
    bool ok = cuda_trace_write(file, &header, sections, job->sizes, job->host);
    free(sections);
    if (fclose(file)) ok = false;
    if (!ok) return false;
    char committed[PATH_MAX];
    snprintf(committed, sizeof(committed), "%s/function-%llu.json", RESULT_DIR, (unsigned long long)job->id);
    if (snprintf(path, sizeof(path), "%s.part", committed) >= sizeof(path)) return false;
    file = fopen(path, "w");
    if (!file) return false;
    fprintf(file, "{\"version\":1,\"launch_id\":%llu,\"pid\":%d,\"device\":%d,"
            "\"context\":\"%p\",\"context_id\":%llu,\"stream\":\"%p\",\"rank\":",
            (unsigned long long)job->id, getpid(), job->loaded->device,
            (void *)job->loaded->context, job->loaded->context_id, (void *)c->hStream);
    nt_json_string(file, getenv("RANK"));
    fputs(",\"kernel\":", file); nt_json_string(file, a->name);
    fputs(",\"artifact\":", file); nt_json_string(file, a->path);
    fprintf(file, ",\"collection_backend\":\"cuda-device-maps-event-async-dtoh-pool\","
            "\"completion_stream\":\"%p\",\"submitter_thread\":\"%lu\"",
            (void *)job->completion_stream, (unsigned long)job->submitter);
    fprintf(file, ",\"status\":\"complete\",\"representative_lane\":0,\"warps\":%zu,"
            "\"metadata_stride\":%zu,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
            "\"shared_mem_bytes\":%u,\"mapped_host\":%s,\"attributes\":[",
            job->warps, a->metadata_stride, c->gridDimX, c->gridDimY, c->gridDimZ,
            c->blockDimX, c->blockDimY, c->blockDimZ, c->sharedMemBytes, "false");
    for (unsigned int i = 0; i < c->numAttrs; ++i) {
        fprintf(file, "%s{\"id\":%d,\"value_hex\":\"", i ? "," : "", c->attrs[i].id);
        const unsigned char *value = (const unsigned char *)&c->attrs[i].value;
        for (size_t j = 0; j < sizeof(c->attrs[i].value); ++j) fprintf(file, "%02x", value[j]);
        fputs("\"}", file);
    }
    fputs("]}\n", file);
    ok = !ferror(file);
    if (fclose(file)) ok = false;
    return ok && rename(path, committed) == 0;
}

static bool nt_journal_submission(nt_job *job) {
    if (!nt_submissions) return false;
    flockfile(nt_submissions);
    fprintf(nt_submissions, "{\"version\":1,\"launch_id\":%llu,\"status\":\"submitted\",\"kernel\":",
            (unsigned long long)job->id);
    nt_json_string(nt_submissions, job->loaded->artifact->name);
    fputs("}\n", nt_submissions);
    bool ok = fflush(nt_submissions) == 0 && !ferror(nt_submissions);
    funlockfile(nt_submissions);
    return ok;
}

static void nt_job_failure(nt_job *job, const char *reason) {
    FILE *file = nt_coverage ? nt_coverage : event_log;
    flockfile(file);
    fprintf(file, "{\"version\":1,\"launch_id\":%llu,\"status\":\"coverage_gap\","
            "\"submitted\":%s,\"cuda_result\":%d,\"kernel\":",
            (unsigned long long)job->id, job->submitted ? "true" : "false", job->launch_result);
    nt_json_string(file, job->loaded->artifact->name);
    fputs(",\"reason\":", file); nt_json_string(file, reason);
    fputs("}\n", file); fflush(file); funlockfile(file);
}

/* Caller holds nt_mutex; errors belong to a context generation and watermark. */
static void nt_pool_error(nt_pool *pool, uint64_t sequence) {
    ++nt_stats.errors;
    if (sequence < pool->first_error_sequence) pool->first_error_sequence = sequence;
}

/* All queue transitions happen under nt_mutex; CUDA and output never do. */
static void nt_export_job(nt_job *job) {
    job->state = NT_EXPORT;
    job->next = NULL;
    *nt_export_tail = job;
    nt_export_tail = &job->next;
    pthread_cond_broadcast(&nt_changed);
}

static void *nt_writer_main(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&nt_mutex);
        while (!nt_journals && !nt_exports && !nt_gap_count && !nt_gap_overflow && !nt_writer_stop)
            pthread_cond_wait(&nt_changed, &nt_mutex);
        nt_job *journal = nt_journals;
        nt_job *job = journal ? NULL : nt_exports;
        nt_gap_record gap = {0};
        uint64_t overflow = 0;
        if (journal) {
            nt_journals = journal->journal_next;
            if (!nt_journals) nt_journal_tail = &nt_journals;
        } else if (job) {
            nt_exports = job->next;
            if (!nt_exports) nt_export_tail = &nt_exports;
            job->state = NT_WRITING;
        } else if (nt_gap_count) {
            gap = nt_gaps[nt_gap_head];
            nt_gap_head = (nt_gap_head + 1) % NT_GAPS;
            --nt_gap_count;
        } else if (nt_gap_overflow) {
            overflow = nt_gap_overflow;
            nt_gap_overflow = 0;
        } else if (nt_writer_stop) { pthread_mutex_unlock(&nt_mutex); break; }
        pthread_mutex_unlock(&nt_mutex);
        if (journal) {
            bool ok = nt_journal_submission(journal);
            pthread_mutex_lock(&nt_mutex);
            if (!ok && !journal->failure) journal->failure = "submission-journal-failed";
            pthread_mutex_unlock(&nt_mutex);
        } else if (job) {
            if (!job->failure && !nt_write_job(job)) job->failure = "trace-write-failed";
            if (job->failure) nt_job_failure(job, job->failure);
            pthread_mutex_lock(&nt_mutex);
            if (job->failure) nt_pool_error(job->pool, job->sequence);
            else ++nt_stats.completed;
            --nt_stats.active;
            nt_job **link = &nt_active;
            while (*link && *link != job) link = &(*link)->active_next;
            if (*link) *link = job->active_next;
            --job->loaded->users;
            job->loaded = NULL;
            if (job->staging->state != 3) job->staging->state = 0;
            job->state = job->quarantined ? NT_QUARANTINED : NT_FREE;
            pthread_cond_broadcast(&nt_changed);
            pthread_mutex_unlock(&nt_mutex);
        } else {
            FILE *file = nt_coverage ? nt_coverage : event_log;
            if (!file) continue;
            flockfile(file);
            fprintf(file, "{\"version\":1,\"launch_id\":%llu,\"status\":\"coverage_gap\","
                          "\"context\":\"%p\",\"function\":\"%p\",\"kernel\":",
                    (unsigned long long)gap.id, (void *)gap.context, (void *)gap.function);
            nt_json_string(file, gap.kernel[0] ? gap.kernel : NULL);
            fputs(",\"reason\":", file);
            nt_json_string(file, overflow ? "coverage-queue-overflow" : gap.reason);
            fprintf(file, ",\"count\":%llu}\n", (unsigned long long)(overflow ? overflow : 1));
            fflush(file);
            funlockfile(file);
        }
    }
    return NULL;
}

static void nt_start_writer(void) {
    nt_writer_started = pthread_create(&nt_writer, NULL, nt_writer_main, NULL) == 0;
}

static void nt_quarantine_job(nt_job *job, const char *reason, bool kernel_uncertain) {
    pthread_mutex_lock(&nt_mutex);
    job->failure = reason;
    job->quarantined = true;
    job->slab->state = job->staging->state = 3;
    if (kernel_uncertain) job->slab->unproven_kernel = job->loaded->quarantined = true;
    nt_export_job(job);
    pthread_mutex_unlock(&nt_mutex);
}

static void nt_poll_job(nt_job *job, enum nt_job_state state) {
    CUresult result = real_cuEventQuery(state == NT_KERNEL_PENDING ? job->event : job->copy_event);
    if (result == CUDA_ERROR_NOT_READY) {
        pthread_mutex_lock(&nt_mutex);
        job->state = state;
        pthread_mutex_unlock(&nt_mutex);
        return;
    }
    if (result != CUDA_SUCCESS) {
        nt_quarantine_job(job, "completion-error", state == NT_KERNEL_PENDING);
        return;
    }
    nt_pool *pool = job->pool;
    if (state == NT_KERNEL_PENDING) {
        result = real_cuStreamWaitEvent(pool->stream, job->event, 0);
        if (result == CUDA_SUCCESS)
            result = real_cuMemcpyDtoHAsync_v2(job->staging->pointer, job->slab->pointer, job->bytes, pool->stream);
        if (result == CUDA_SUCCESS) result = real_cuEventRecord(job->copy_event, pool->stream);
        if (result != CUDA_SUCCESS) { nt_quarantine_job(job, "trace-copy-failed", false); return; }
        pthread_mutex_lock(&nt_mutex);
        job->state = NT_COPY_PENDING;
        pthread_mutex_unlock(&nt_mutex);
    } else {
        nt_device_slab *slab = job->slab;
        result = real_cuMemsetD32Async(slab->pointer, 0, slab->capacity / 4, pool->stream);
        if (result == CUDA_SUCCESS) result = real_cuEventRecord(slab->ready, pool->stream);
        pthread_mutex_lock(&nt_mutex);
        slab->state = result == CUDA_SUCCESS ? 2 : 3;
        slab->sequence = job->sequence;
        if (result != CUDA_SUCCESS) job->failure = "buffer-reset-failed";
        job->copied = true;
        nt_export_job(job);
        pthread_mutex_unlock(&nt_mutex);
    }
}

static void *nt_collection_main(void *argument) {
    nt_pool *pool = argument;
    CUcontext popped;
    bool current = real_cuCtxPushCurrent_v2(pool->context) == CUDA_SUCCESS;
    for (;;) {
        pthread_mutex_lock(&nt_mutex);
        bool stop = pool->stopped;
        nt_capacity *classes = pool->classes;
        pthread_mutex_unlock(&nt_mutex);
        if (stop) break;
        for (nt_capacity *c = classes; c; c = c->next) {
            pthread_mutex_lock(&nt_mutex);
            bool preparing = c->preparing;
            pthread_mutex_unlock(&nt_mutex);
            if (preparing) continue;
            for (size_t i = 0; i < c->count; ++i) {
                nt_job *job = &c->jobs[i];
                nt_device_slab *slab = &c->device[i];
                pthread_mutex_lock(&nt_mutex);
                enum nt_job_state state = job->state;
                bool poll = state == NT_KERNEL_PENDING || state == NT_COPY_PENDING;
                if (poll) job->state = NT_WORKING;
                bool reset = slab->state == 2;
                pthread_mutex_unlock(&nt_mutex);
                if (poll) {
                    if (current) nt_poll_job(job, state);
                    else nt_quarantine_job(job, "collection-context-unavailable", true);
                }
                if (reset) {
                    CUresult result = current ? real_cuEventQuery(slab->ready) : CUDA_ERROR_INVALID_CONTEXT;
                    pthread_mutex_lock(&nt_mutex);
                    if (result == CUDA_SUCCESS) slab->state = 0;
                    else if (result != CUDA_ERROR_NOT_READY) {
                        slab->state = 3;
                        nt_pool_error(pool, slab->sequence);
                    }
                    pthread_cond_broadcast(&nt_changed);
                    pthread_mutex_unlock(&nt_mutex);
                }
            }
        }
        /* Poll every job, never block behind an unfinished event. */
        struct timespec until;
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_nsec += 100000;
        if (until.tv_nsec >= 1000000000) { ++until.tv_sec; until.tv_nsec -= 1000000000; }
        pthread_mutex_lock(&nt_mutex);
        if (!pool->stopped) pthread_cond_timedwait(&nt_changed, &nt_mutex, &until);
        pthread_mutex_unlock(&nt_mutex);
    }
    if (current) real_cuCtxPopCurrent_v2(&popped);
    return NULL;
}

static uint64_t nt_clock_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000ull + now.tv_nsec;
}

/* Only the first encounter of a capacity class performs cold preparation. */
static nt_pool *nt_pool_get(nt_loaded *loaded, size_t bytes) {
    pthread_once(&nt_writer_once, nt_start_writer);
    if (!nt_writer_started) return NULL;
    size_t capacity = 256;
    while (capacity < bytes && capacity <= SIZE_MAX / 2) capacity *= 2;
    if (capacity < bytes || capacity > SIZE_MAX / 2) return NULL;
    uint64_t started = nt_clock_ns();
    pthread_mutex_lock(&nt_mutex);
    nt_pool *pool = nt_pools;
    while (pool && !(pool->context == loaded->context && pool->generation == loaded->context_id && !pool->stopped))
        pool = pool->next;
    if (nt_stopping || loaded->closing || (pool && (pool->closing || pool->preparing))) {
        pthread_mutex_unlock(&nt_mutex); return NULL;
    }
    bool fresh = !pool;
    if (fresh) {
        pool = calloc(1, sizeof(*pool));
        if (!pool) { pthread_mutex_unlock(&nt_mutex); return NULL; }
        pool->context = loaded->context; pool->generation = loaded->context_id;
        pool->first_error_sequence = UINT64_MAX;
        pool->preparing = true; pool->next = nt_pools; nt_pools = pool;
    }
    nt_capacity *c = pool->classes;
    while (c && c->bytes != capacity) c = c->next;
    if (c) { pthread_mutex_unlock(&nt_mutex); return pool; }
    c = calloc(1, sizeof(*c));
    if (!c) { pool->preparing = false; if (fresh) pool->closing = true; pthread_mutex_unlock(&nt_mutex); return NULL; }
    c->bytes = capacity; c->preparing = true;
    c->count = nt_inflight <= nt_budget ? (nt_budget - nt_inflight) / (2 * capacity) : 0;
    if (c->count > nt_prewarm) c->count = nt_prewarm;
    nt_inflight += c->count * 2 * capacity;
    if (nt_inflight > nt_stats.resident_high_water) nt_stats.resident_high_water = nt_inflight;
    c->next = pool->classes; pool->classes = c;
    pthread_mutex_unlock(&nt_mutex);
    bool ok = !fresh || real_cuStreamCreate(&pool->stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS;
    c->device = calloc(c->count, sizeof(*c->device));
    c->host = calloc(c->count, sizeof(*c->host));
    c->jobs = calloc(c->count, sizeof(*c->jobs));
    if (!c->device || !c->host || !c->jobs) {
        free(c->device); free(c->host); free(c->jobs);
        c->device = NULL; c->host = NULL; c->jobs = NULL;
        pthread_mutex_lock(&nt_mutex);
        nt_inflight -= c->count * 2 * capacity;
        c->count = 0;
        pthread_mutex_unlock(&nt_mutex);
        ok = false;
    }
    uint64_t devices = 0, hosts = 0, events = 0;
    for (size_t i = 0; i < c->count; ++i) {
        nt_device_slab *d = &c->device[i];
        nt_host_slab *h = &c->host[i];
        nt_job *j = &c->jobs[i];
        d->capacity = h->capacity = d->charge = h->charge = capacity;
        d->state = h->state = 3;
        j->state = NT_QUARANTINED;
        j->pool = pool;
        j->device = j->device_storage; j->host = j->host_storage;
        j->sizes = j->size_storage; j->attributes = j->attribute_storage;
        if (!ok) continue;
        if (real_cuMemAlloc_v2(&d->pointer, capacity) != CUDA_SUCCESS) { ok = false; continue; }
        ++devices;
        if (real_cuMemAllocHost_v2(&h->pointer, capacity) != CUDA_SUCCESS) { ok = false; continue; }
        ++hosts;
        CUevent *handles[] = {&d->ready, &j->event, &j->copy_event};
        for (size_t k = 0; k < 3; ++k) {
            if (real_cuEventCreate(handles[k], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) { ok = false; break; }
            ++events;
        }
        if (!ok) continue;
        ok = real_cuMemsetD32Async(d->pointer, 0, capacity / 4, pool->stream) == CUDA_SUCCESS &&
             real_cuEventRecord(d->ready, pool->stream) == CUDA_SUCCESS &&
             real_cuEventSynchronize(d->ready) == CUDA_SUCCESS;
        if (ok) { d->state = h->state = 0; j->state = NT_FREE; }
    }
    bool worker_ok = !fresh || (pool->stream && pthread_create(&pool->worker, NULL, nt_collection_main, pool) == 0);
    pthread_mutex_lock(&nt_mutex);
    if (fresh) pool->worker_started = worker_ok;
    pool->preparing = false;
    if (!worker_ok) pool->closing = true;
    c->preparing = false;
    nt_stats.device_allocations += devices; nt_stats.host_allocations += hosts;
    nt_stats.event_allocations += events; nt_stats.cold_ns += nt_clock_ns() - started;
    if (!ok || !worker_ok) nt_pool_error(pool, 0);
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
    return worker_ok ? pool : NULL;
}

static nt_job *nt_acquire(nt_pool *pool, nt_loaded *loaded, size_t bytes, const char **reason) {
    nt_device_slab *device = NULL;
    nt_host_slab *host = NULL;
    nt_job *job = NULL;
    pthread_mutex_lock(&nt_mutex);
    if (pool && !pool->closing && !loaded->closing && !nt_stopping) {
        for (nt_capacity *c = pool->classes; c; c = c->next) {
            if (c->preparing) continue;
            for (size_t i = 0; i < c->count; ++i) {
                if (c->bytes >= bytes) {
                    if (!c->device[i].state && (!device || device->capacity > c->bytes)) device = &c->device[i];
                    if (!c->host[i].state && (!host || host->capacity > c->bytes)) host = &c->host[i];
                }
                if (!job && c->jobs[i].state == NT_FREE) job = &c->jobs[i];
            }
        }
    }
    if (!device || !host || !job) {
        ++nt_stats.misses;
        *reason = !pool ? "pool-preparation-unavailable" : !device ? "pool-device-exhausted" :
                  !host ? "pool-host-exhausted" : "pool-job-exhausted";
        pthread_mutex_unlock(&nt_mutex); return NULL;
    }
    ++nt_stats.hits;
    device->state = host->state = 1;
    job->slab = device; job->staging = host; job->loaded = loaded;
    job->context = pool->context; job->state = NT_RESERVED;
    job->failure = NULL; job->submitted = job->copied = job->quarantined = false;
    job->sequence = 0; job->id = ++nt_launch_serial;
    pthread_mutex_unlock(&nt_mutex);
    return job;
}

/* Caller holds nt_mutex. Future submissions never extend this watermark. */
static CUresult nt_flush_watermark(CUcontext context, unsigned long long generation, uint64_t watermark) {
    for (;;) {
        bool busy = false;
        for (nt_job *j = nt_active; j; j = j->active_next)
            if (j->context == context && j->pool->generation == generation && j->sequence <= watermark) busy = true;
        for (nt_pool *p = nt_pools; p; p = p->next) {
            if (p->context != context || p->generation != generation || p->stopped) continue;
            for (nt_capacity *c = p->classes; c; c = c->next) {
                if (c->preparing) continue;
                for (size_t i = 0; i < c->count; ++i)
                    if (c->device[i].state == 2 && c->device[i].sequence <= watermark) busy = true;
            }
        }
        if (!busy) break;
        pthread_cond_wait(&nt_changed, &nt_mutex);
    }
    for (nt_pool *p = nt_pools; p; p = p->next)
        if (p->context == context && p->generation == generation && p->first_error_sequence <= watermark)
            return CUDA_ERROR_UNKNOWN;
    return CUDA_SUCCESS;
}

/* Explicit flush waits only for the accepted-job watermark, including resets. */
CUresult neutrinoFunctionFlush(void) {
    if (!nt_enabled) return CUDA_SUCCESS;
    CUcontext context = nt_current();
    unsigned long long generation;
    if (!context || real_cuCtxGetId(context, &generation) != CUDA_SUCCESS) return CUDA_ERROR_INVALID_CONTEXT;
    pthread_mutex_lock(&nt_mutex);
    CUresult result = nt_flush_watermark(context, generation, nt_stats.accepted);
    pthread_mutex_unlock(&nt_mutex);
    return result;
}

/* Returns required bytes including NUL. A short destination receives valid NUL-terminated text. */
size_t neutrinoFunctionGetStats(char *destination, size_t size) {
    char json[8192];
    pthread_mutex_lock(&nt_mutex);
    size_t kernels = 0, copies = 0, exports = 0, resetting = 0, ready_device = 0, ready_host = 0, quarantined = 0;
    for (nt_job *j = nt_active; j; j = j->active_next) {
        kernels += j->state == NT_KERNEL_PENDING;
        copies += j->state == NT_COPY_PENDING || j->state == NT_WORKING;
        exports += j->state == NT_EXPORT || j->state == NT_WRITING;
    }
    for (nt_pool *p = nt_pools; p; p = p->next) {
        if (p->stopped) continue;
        for (nt_capacity *c = p->classes; c; c = c->next) {
            if (c->preparing) continue;
            for (size_t i = 0; i < c->count; ++i) {
                ready_device += c->device[i].state == 0; ready_host += c->host[i].state == 0;
                resetting += c->device[i].state == 2;
                quarantined += c->device[i].state == 3 || c->host[i].state == 3;
            }
        }
    }
    int n = snprintf(json, sizeof(json), "{\"backend\":\"cuda-device-maps-event-async-dtoh-pool\","
        "\"accepted\":%llu,\"completed\":%llu,\"skipped\":%llu,\"pool_hits\":%llu,\"pool_misses\":%llu,"
        "\"errors\":%llu,\"device_allocations\":%llu,\"host_allocations\":%llu,\"event_allocations\":%llu,"
        "\"cold_preparation_ns\":%llu,\"resident_bytes\":%zu,\"budget_bytes\":%zu,\"prewarm\":%u,"
        "\"resident_high_water\":%zu,\"active\":%zu,\"high_water\":%zu,\"pending_kernels\":%zu,"
        "\"pending_copies\":%zu,\"pending_exports\":%zu,\"resetting\":%zu,\"ready_device\":%zu,"
        "\"ready_host\":%zu,\"quarantined_slots\":%zu,\"skip_reasons\":{",
        (unsigned long long)nt_stats.accepted, (unsigned long long)nt_stats.completed,
        (unsigned long long)nt_stats.skipped, (unsigned long long)nt_stats.hits, (unsigned long long)nt_stats.misses,
        (unsigned long long)nt_stats.errors, (unsigned long long)nt_stats.device_allocations,
        (unsigned long long)nt_stats.host_allocations, (unsigned long long)nt_stats.event_allocations,
        (unsigned long long)nt_stats.cold_ns, nt_inflight, nt_budget, nt_prewarm, nt_stats.resident_high_water,
        nt_stats.active, nt_stats.high_water, kernels, copies, exports, resetting, ready_device, ready_host, quarantined);
    for (size_t i = 0; i < 64 && nt_skip_reasons[i].reason && n > 0 && (size_t)n < sizeof(json) - 128; ++i)
        n += snprintf(json + n, sizeof(json) - n, "%s\"%s\":%llu", i ? "," : "", nt_skip_reasons[i].reason,
                      (unsigned long long)nt_skip_reasons[i].count);
    if (n > 0 && (size_t)n < sizeof(json) - 3) { json[n++] = '}'; json[n++] = '}'; json[n] = 0; }
    pthread_mutex_unlock(&nt_mutex);
    size_t required = strlen(json) + 1;
    if (destination && size) { size_t count = required < size ? required : size; memcpy(destination, json, count); destination[count - 1] = 0; }
    return required;
}

static void nt_drain(CUcontext context) {
    pthread_mutex_lock(&nt_mutex);
    for (;;) {
        bool busy = false;
        for (nt_loaded *l = nt_loaded_images; l; l = l->next)
            if ((!context || l->context == context) && l->users) busy = true;
        if (!busy) break;
        pthread_cond_wait(&nt_changed, &nt_mutex);
    }
    pthread_mutex_unlock(&nt_mutex);
}

/* Binding ancestry is inspected only under nt_mutex. */
static bool nt_descends(uint64_t id, uint64_t root) {
    while (id) {
        if (id == root) return true;
        nt_binding *b = nt_bindings;
        while (b && b->id != id) b = b->next;
        if (!b) break;
        id = b->parent;
    }
    return false;
}

static bool nt_scope(nt_loaded *loaded, uint64_t root, CUcontext context) {
    return root ? nt_descends(loaded->binding_id, root) : (!context || loaded->context == context);
}

static bool nt_teardown_ok(CUresult result) {
    if (result == CUDA_SUCCESS) return true;
    pthread_mutex_lock(&nt_mutex);
    ++nt_stats.errors;
    pthread_mutex_unlock(&nt_mutex);
    return false;
}

static void nt_pool_cleanup(nt_pool *pool, bool destroyed) {
    CUcontext popped;
    bool current = !destroyed && nt_teardown_ok(real_cuCtxPushCurrent_v2(pool->context));
    bool safe = current && (!pool->stream || nt_teardown_ok(real_cuStreamSynchronize(pool->stream)));
    size_t released = 0;
    for (nt_capacity *c = pool->classes; c; c = c->next) {
        for (size_t i = 0; i < c->count; ++i) {
            nt_device_slab *d = &c->device[i];
            nt_host_slab *h = &c->host[i];
            nt_job *j = &c->jobs[i];
            if (destroyed) { d->pointer = 0; d->ready = j->event = j->copy_event = NULL; }
            if (safe && !d->unproven_kernel && d->pointer && nt_teardown_ok(real_cuMemFree_v2(d->pointer))) d->pointer = 0;
            if ((destroyed || safe) && h->pointer && nt_teardown_ok(real_cuMemFreeHost(h->pointer))) h->pointer = NULL;
            if (safe && !d->unproven_kernel && d->ready && nt_teardown_ok(real_cuEventDestroy_v2(d->ready))) d->ready = NULL;
            if (safe && j->state != NT_QUARANTINED) {
                if (j->event && nt_teardown_ok(real_cuEventDestroy_v2(j->event))) j->event = NULL;
                if (j->copy_event && nt_teardown_ok(real_cuEventDestroy_v2(j->copy_event))) j->copy_event = NULL;
            }
            if (!d->pointer) { released += d->charge; d->charge = 0; }
            if (!h->pointer) { released += h->charge; h->charge = 0; }
        }
        bool empty = true;
        for (size_t i = 0; i < c->count; ++i)
            if (c->device[i].pointer || c->device[i].ready || c->host[i].pointer ||
                c->jobs[i].event || c->jobs[i].copy_event) empty = false;
        if (empty) {
            free(c->device); free(c->host); free(c->jobs);
            c->device = NULL; c->host = NULL; c->jobs = NULL; c->count = 0;
        }
    }
    if (safe && pool->stream && nt_teardown_ok(real_cuStreamDestroy_v2(pool->stream))) pool->stream = NULL;
    if (destroyed) pool->stream = NULL;
    if (current) real_cuCtxPopCurrent_v2(&popped);
    pthread_mutex_lock(&nt_mutex);
    nt_inflight -= released;
    pthread_mutex_unlock(&nt_mutex);
}

static void nt_stop_pools(CUcontext context) {
    for (;;) {
        pthread_mutex_lock(&nt_mutex);
        nt_pool *pool = nt_pools;
        while (pool && ((context && pool->context != context) || pool->stopped || pool->joining)) pool = pool->next;
        if (!pool) { pthread_mutex_unlock(&nt_mutex); break; }
        pool->joining = true; pool->stopped = true;
        pthread_cond_broadcast(&nt_changed);
        pthread_mutex_unlock(&nt_mutex);
        if (pool->worker_started) pthread_join(pool->worker, NULL);
        nt_pool_cleanup(pool, false);
        pthread_mutex_lock(&nt_mutex);
        pool->joining = false;
        pthread_cond_broadcast(&nt_changed);
        pthread_mutex_unlock(&nt_mutex);
    }
    pthread_mutex_lock(&nt_mutex);
    for (;;) {
        bool joining = false;
        for (nt_pool *p = nt_pools; p; p = p->next)
            if ((!context || p->context == context) && p->joining) joining = true;
        if (!joining) break;
        pthread_cond_wait(&nt_changed, &nt_mutex);
    }
    pthread_mutex_unlock(&nt_mutex);
}

static void nt_retire(uint64_t root, CUcontext context) {
    pthread_mutex_lock(&nt_mutex);
    if (!root) for (nt_pool *p = nt_pools; p; p = p->next)
        if (!context || p->context == context) p->closing = true;
    for (nt_binding *b = nt_bindings; b; b = b->next)
        if (root ? nt_descends(b->id, root) : (!context || b->context == context)) b->closing = true;
    for (;;) {
        bool busy = false;
        for (nt_loaded *l = nt_loaded_images; l; l = l->next) {
            if (!nt_scope(l, root, context)) continue;
            l->closing = true;
            if (l->users) busy = true;
        }
        if (!busy) break;
        pthread_cond_wait(&nt_changed, &nt_mutex);
    }
    nt_loaded *retired = NULL, **link = &nt_loaded_images;
    while (*link) {
        nt_loaded *l = *link;
        if (!nt_scope(l, root, context)) { link = &l->next; continue; }
        *link = l->next; l->next = retired; retired = l;
    }
    pthread_mutex_unlock(&nt_mutex);
    if (!root) nt_stop_pools(context);
    while (retired) {
        nt_loaded *l = retired; retired = l->next;
        CUcontext popped;
        if (real_cuCtxPushCurrent_v2(l->context) == CUDA_SUCCESS) {
            if (l->module && !l->quarantined) real_cuModuleUnload(l->module);
            real_cuCtxPopCurrent_v2(&popped);
        }
        free(l);
    }
}

static uint64_t nt_retire_handle(enum nt_kind kind, void *handle, CUcontext context) {
    if (!nt_enabled) return 0;
    pthread_mutex_lock(&nt_mutex);
    nt_binding *b = nt_find(kind, handle, context);
    uint64_t id = b ? b->id : 0;
    pthread_mutex_unlock(&nt_mutex);
    if (id) nt_retire(id, NULL);
    return id;
}

static void nt_reopen(uint64_t root, CUcontext context) {
    pthread_mutex_lock(&nt_mutex);
    for (nt_binding *b = nt_bindings; b; b = b->next)
        if (root ? nt_descends(b->id, root) : (context && b->context == context)) b->closing = false;
    pthread_mutex_unlock(&nt_mutex);
}

/* A destroyed/reset context proves quarantined GPU accesses have ceased. */
static void nt_reclaim(CUcontext context) {
    pthread_mutex_lock(&nt_mutex);
    nt_pool *pools = nt_pools;
    pthread_mutex_unlock(&nt_mutex);
    for (nt_pool *p = pools; p; p = p->next) {
        pthread_mutex_lock(&nt_mutex);
        if (p->context != context || !p->stopped) { pthread_mutex_unlock(&nt_mutex); continue; }
        while (p->joining) pthread_cond_wait(&nt_changed, &nt_mutex);
        p->joining = true;
        pthread_mutex_unlock(&nt_mutex);
        nt_pool_cleanup(p, true);
        pthread_mutex_lock(&nt_mutex);
        p->joining = false;
        pthread_cond_broadcast(&nt_changed);
        pthread_mutex_unlock(&nt_mutex);
    }
}

static void nt_shutdown(void) {
    if (!nt_enabled) return;
    /* Stop new preparation, but let existing submitters publish their jobs. */
    pthread_mutex_lock(&nt_mutex);
    for (nt_binding *b = nt_bindings; b; b = b->next) b->closing = true;
    nt_stopping = true;
    pthread_mutex_unlock(&nt_mutex);
    nt_retire(0, NULL);
    pthread_mutex_lock(&nt_mutex);
    nt_writer_stop = true;
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
    if (nt_writer_started) { pthread_join(nt_writer, NULL); nt_writer_started = false; }
    else nt_writer_main(NULL); /* Persist bounded skip metadata if worker creation failed. */
    char path[PATH_MAX], json[8192];
    if (snprintf(path, sizeof(path), "%s/pool-stats.json", RESULT_DIR) < sizeof(path)) {
        neutrinoFunctionGetStats(json, sizeof(json));
        FILE *file = fopen(path, "w");
        if (file) { fputs(json, file); fputc('\n', file); fclose(file); }
    }
}

static void nt_preference_set(enum nt_kind kind, void *handle, CUcontext context,
                              CUdevice device, bool shared, int value) {
    if (!nt_enabled) return;
    pthread_mutex_lock(&nt_mutex);
    nt_binding *b = nt_find(kind, handle, context);
    if (b) {
        nt_preference *p = nt_preferences;
        while (p && !(p->binding_id == b->id && p->device == device && p->shared == shared)) p = p->next;
        if (!p) {
            p = calloc(1, sizeof(*p));
            if (p) {
                *p = (nt_preference){.binding_id=b->id, .device=device, .shared=shared, .next=nt_preferences};
                nt_preferences = p;
            }
        }
        if (p) p->value = value;
    }
    pthread_mutex_unlock(&nt_mutex);
}

static bool nt_apply_attributes(CUfunction original, nt_loaded *loaded, const CUlaunchConfig *config,
                                size_t blocks, size_t threads) {
    bool found[2] = {false, false};
    int preference[2] = {0, 0};
    pthread_mutex_lock(&nt_mutex);
    nt_binding *binding = nt_find(NT_FUNCTION, original, loaded->context);
    while (binding) {
        for (nt_preference *p = nt_preferences; p; p = p->next) {
            if (p->binding_id == binding->id && (p->device == -1 || p->device == loaded->device) && !found[p->shared]) {
                found[p->shared] = true; preference[p->shared] = p->value;
            }
        }
        uint64_t parent = binding->parent;
        binding = nt_bindings;
        while (binding && binding->id != parent) binding = binding->next;
    }
    pthread_mutex_unlock(&nt_mutex);
    if (found[0] && real_cuFuncSetCacheConfig(loaded->function, preference[0]) != CUDA_SUCCESS) return false;
    if (found[1] && real_cuFuncSetSharedMemConfig(loaded->function, preference[1]) != CUDA_SUCCESS) return false;
    const CUfunction_attribute mutable[] = {
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
        CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
    };
    size_t cluster[3] = {1, 1, 1};
    for (size_t i = 0; i < sizeof(mutable) / sizeof(*mutable); ++i) {
        int value, replacement;
        if (real_cuFuncGetAttribute(&value, mutable[i], original) != CUDA_SUCCESS ||
            real_cuFuncGetAttribute(&replacement, mutable[i], loaded->function) != CUDA_SUCCESS) return false;
        if (value != replacement && real_cuFuncSetAttribute(loaded->function, mutable[i], value) != CUDA_SUCCESS)
            return false;
        if (i >= 2 && i <= 4 && value > 0) cluster[i - 2] = value;
    }
    bool cooperative = false;
    for (unsigned int i = 0; i < config->numAttrs; ++i) {
        const CUlaunchAttribute *a = &config->attrs[i];
        if (a->id == CU_LAUNCH_ATTRIBUTE_COOPERATIVE) cooperative = a->value.cooperative != 0;
        if (a->id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
            cluster[0] = a->value.clusterDim.x; cluster[1] = a->value.clusterDim.y; cluster[2] = a->value.clusterDim.z;
        }
    }
    if (!cooperative) return true;
    size_t cluster_size;
    if (!nt_multiply(cluster[0], cluster[1], &cluster_size) ||
        !nt_multiply(cluster_size, cluster[2], &cluster_size) || !cluster_size) return false;
    if (cluster_size > 1) {
        int active;
        return real_cuOccupancyMaxActiveClusters(&active, loaded->function, config) == CUDA_SUCCESS &&
               active > 0 && blocks % cluster_size == 0 && blocks / cluster_size <= (size_t)active;
    }
    int active, sms;
    return real_cuOccupancyMaxActiveBlocksPerMultiprocessor(&active, loaded->function, (int)threads,
               config->sharedMemBytes) == CUDA_SUCCESS &&
           real_cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, loaded->device) == CUDA_SUCCESS &&
           active > 0 && sms > 0 && blocks <= (size_t)active * sms;
}

/* true means a replacement was submitted, including a failed submission. The
 * caller may forward only on false; there is never a second application launch. */
static bool nt_launch(const CUlaunchConfig *config, CUfunction function, void **params, void **extra,
                      bool extended, bool per_thread, CUresult *result) {
    const char *reason = "untracked-image-or-function";
    nt_view view = {0};
    nt_loaded *loaded = NULL;
    nt_job *job = NULL;
    void **arguments = NULL;
    CUlaunchConfig copy;
    size_t blocks, threads, warps, payload = 0, bytes;
    view = nt_lookup(function, nt_current());
    const char *include = getenv("NEUTRINO_KERNEL"), *exclude = getenv("NEUTRINO_FILTER");
    if ((include && *include) || (exclude && *exclude)) {
        const char *name = view.name;
        if (!name && real_cuFuncGetName && real_cuFuncGetName(&name, function) != CUDA_SUCCESS) name = NULL;
        if (!name && real_cuKernelGetName && real_cuKernelGetName(&name, (CUkernel)function) != CUDA_SUCCESS) name = NULL;
        if (name && ((include && *include && !nt_name_matches(include, name)) || nt_name_matches(exclude, name))) {
            nt_view_release(&view); return false;
        }
    }
    if (extra) { reason = "packed-extra-launch"; goto fallback; }
    if (!config || !nt_read_memory(&copy, config, sizeof(copy))) { reason = "invalid-launch-config"; goto fallback; }
    CUstream stream = copy.hStream ? copy.hStream : (per_thread ? CU_STREAM_PER_THREAD : CU_STREAM_LEGACY);
    CUstreamCaptureStatus capture;
    if (!real_cuStreamIsCapturing_ptsz || real_cuStreamIsCapturing_ptsz(stream, &capture) != CUDA_SUCCESS) {
        reason = "capture-status-unavailable"; goto fallback;
    }
    if (capture != CU_STREAM_CAPTURE_STATUS_NONE) { reason = "graph-capture"; goto fallback; }
    if (!view.image || !view.name) goto fallback;
    nt_artifact *artifact = nt_artifact_get(&view);
    if (artifact && artifact->status == 2) { nt_view_release(&view); return false; }
    if (!artifact || artifact->status != 1) { reason = "function-compilation-unsupported"; goto fallback; }
    unsigned long long context_id;
    if (real_cuCtxGetId(view.context, &context_id) != CUDA_SUCCESS ||
        !(loaded = nt_loaded_get(&view, artifact, context_id)) || loaded->status != 1) {
        reason = "replacement-load-failed"; goto fallback;
    }
    if (!nt_multiply(copy.gridDimX, copy.gridDimY, &blocks) || !nt_multiply(blocks, copy.gridDimZ, &blocks) ||
        !nt_multiply(copy.blockDimX, copy.blockDimY, &threads) || !nt_multiply(threads, copy.blockDimZ, &threads) ||
        !blocks || !threads || threads > 1024 || !nt_multiply((threads + 31) / 32, blocks, &warps)) {
        reason = "invalid-launch-dimensions"; goto fallback;
    }
    for (unsigned int i = 0; i < artifact->nmaps; ++i) {
        size_t map_bytes;
        if (!nt_multiply(artifact->maps[i].size * artifact->maps[i].cap, warps, &map_bytes) ||
            map_bytes > SIZE_MAX - 255 || payload > SIZE_MAX - ((map_bytes + 255) & ~(size_t)255)) {
            reason = "buffer-size-overflow"; goto fallback;
        }
        payload += (map_bytes + 255) & ~(size_t)255;
    }
    bytes = payload;
    if (copy.numAttrs > NT_MAX_ATTRS) { reason = "invalid-launch-attributes"; goto fallback; }
    nt_pool *pool = nt_pool_get(loaded, bytes);
    job = nt_acquire(pool, loaded, bytes, &reason);
    if (!job) goto fallback;
    job->config = copy;
    if (copy.numAttrs && !nt_read_memory(job->attributes, copy.attrs, copy.numAttrs * sizeof(CUlaunchAttribute))) {
        reason = "invalid-launch-attributes"; goto fallback;
    }
    job->config.attrs = copy.numAttrs ? job->attributes : NULL;
    if (!nt_apply_attributes(function, loaded, &job->config, blocks, threads)) {
        reason = "replacement-attributes-or-residency"; goto fallback;
    }
    arguments = job->arguments;
    job->nmaps = artifact->nmaps;
    if (artifact->nparams && !nt_read_memory(arguments, params, artifact->nparams * sizeof(void *))) {
        reason = "invalid-kernel-parameters"; goto fallback;
    }
    job->warps = warps; job->bytes = bytes; job->payload_bytes = payload;
    job->completion_stream = stream; job->submitter = pthread_self();
    size_t offset = 0;
    for (unsigned int i = 0; i < artifact->nmaps; ++i) {
        job->sizes[i] = warps * artifact->maps[i].size * artifact->maps[i].cap;
        job->device[i] = job->slab->pointer + offset;
        job->host[i] = (char *)job->staging->pointer + offset;
        offset += (job->sizes[i] + 255) & ~(size_t)255;
        arguments[artifact->nparams + i] = &job->device[i];
    }
    job->submitted = true;
    if (extended)
        *result = per_thread ? real_cuLaunchKernelEx_ptsz(&job->config, loaded->function, arguments, NULL) :
                              real_cuLaunchKernelEx(&job->config, loaded->function, arguments, NULL);
    else {
        CUresult (*launch)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int, CUstream, void **, void **) =
            per_thread ? real_cuLaunchKernel_ptsz : real_cuLaunchKernel;
        *result = launch(loaded->function, copy.gridDimX, copy.gridDimY, copy.gridDimZ,
                         copy.blockDimX, copy.blockDimY, copy.blockDimZ, copy.sharedMemBytes,
                         copy.hStream, arguments, NULL);
    }
    job->launch_result = *result;
    /* Record on the submitting thread: PTDS identity cannot be recovered by a worker. */
    CUresult recorded = real_cuEventRecord_ptsz(job->event, stream);
    pthread_mutex_lock(&nt_mutex);
    job->sequence = ++nt_stats.accepted;
    ++nt_stats.active;
    if (nt_stats.active > nt_stats.high_water) nt_stats.high_water = nt_stats.active;
    job->active_next = nt_active; nt_active = job;
    job->journal_next = NULL;
    *nt_journal_tail = job; nt_journal_tail = &job->journal_next;
    if (*result != CUDA_SUCCESS) job->failure = "replacement-launch-failed-no-retry";
    if (recorded != CUDA_SUCCESS) {
        job->failure = "completion-record-failed";
        job->slab->unproven_kernel = job->quarantined = loaded->quarantined = true;
        job->slab->state = job->staging->state = 3;
        nt_export_job(job);
    } else job->state = NT_KERNEL_PENDING;
    pthread_cond_broadcast(&nt_changed);
    pthread_mutex_unlock(&nt_mutex);
    nt_view_release(&view);
    return true;
fallback:
    if (job) {
        pthread_mutex_lock(&nt_mutex);
        job->slab->state = job->staging->state = 0;
        job->state = NT_FREE; job->loaded = NULL;
        pthread_mutex_unlock(&nt_mutex);
    }
    if (loaded) nt_loaded_release(loaded);
    nt_view_release(&view);
    nt_gap(function, reason);
    return false;
}

static void nt_init(void) {
    const char *mode = getenv("NEUTRINO_FUNCTION_PROBE_ABI");
    nt_enabled = mode && !strcmp(mode, "1");
    if (!nt_enabled) return;
    NEUTRINO_BENCHMARK = 0;
    NEUTRINO_MEMUSAGE = 0;
    DYNAMIC = 0;
    const char *budget = getenv("NEUTRINO_BUFFER_BYTES");
    if (budget && *budget) {
        char *end;
        errno = 0;
        unsigned long long bytes = strtoull(budget, &end, 10);
        nt_budget = !errno && !*end && *budget != '-' && bytes <= SIZE_MAX ? (size_t)bytes : 0;
    }
    const char *prewarm = getenv("NEUTRINO_POOL_PREWARM");
    if (prewarm && *prewarm) {
        char *end;
        unsigned long count = strtoul(prewarm, &end, 10);
        nt_prewarm = !*end && count > 0 && count <= 1024 ? count : 8;
    }
    atexit(nt_shutdown);
    char *absolute = realpath(RESULT_DIR, NULL);
    if (absolute) { free(RESULT_DIR); RESULT_DIR = absolute; }
    absolute = realpath(KERNEL_DIR, NULL);
    if (absolute) { free(KERNEL_DIR); KERNEL_DIR = absolute; }
    size_t length = strlen(RESULT_DIR) + 32;
    char *path = malloc(length);
    if (path) {
        snprintf(path, length, "%s/coverage.jsonl", RESULT_DIR);
        nt_coverage = fopen(path, "a");
        snprintf(path, length, "%s/launches.jsonl", RESULT_DIR);
        nt_submissions = fopen(path, "a");
        free(path);
    }
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *path,
        CUjit_option *jit, void **jit_values, unsigned int njit,
        CUlibraryOption *options, void **values, unsigned int noptions) {
    if (!shared_lib) init();
    nt_image *image = nt_snapshot_file(path);
    CUresult result = real_cuLibraryLoadFromFile(library, path, jit, jit_values, njit, options, values, noptions);
    nt_register(NT_LIBRARY, result == CUDA_SUCCESS ? *library : NULL, NULL, image, result);
    return result;
}

CUresult cuLibraryEnumerateKernels(CUkernel *kernels, unsigned int count, CUlibrary library) {
    if (!shared_lib) init();
    CUresult result = real_cuLibraryEnumerateKernels(kernels, count, library);
    unsigned int actual = 0;
    if (nt_enabled && result == CUDA_SUCCESS &&
        real_cuLibraryGetKernelCount(&actual, library) == CUDA_SUCCESS) {
        for (unsigned int i = 0; i < count && i < actual; ++i) {
            const char *name = NULL;
            if (real_cuKernelGetName(&name, kernels[i]) == CUDA_SUCCESS)
                nt_bind(NT_LIBRARY, library, NULL, NT_KERNEL, kernels[i], NULL, name);
        }
    }
    return result;
}

CUresult cuLibraryUnload(CUlibrary library) {
    if (!shared_lib) init();
    uint64_t id = nt_retire_handle(NT_LIBRARY, library, NULL);
    CUresult result = real_cuLibraryUnload(library);
    if (result == CUDA_SUCCESS) nt_forget(NT_LIBRARY, library, NULL);
    else if (id) nt_reopen(id, NULL);
    return result;
}

CUresult cuModuleUnload(CUmodule module) {
    if (!shared_lib) init();
    CUcontext context = nt_enabled ? nt_current() : NULL;
    uint64_t id = nt_retire_handle(NT_MODULE, module, context);
    CUresult result = real_cuModuleUnload(module);
    if (result == CUDA_SUCCESS) nt_forget(NT_MODULE, module, context);
    else if (id) nt_reopen(id, NULL);
    return result;
}

CUresult cuCtxDestroy_v2(CUcontext context) {
    if (!shared_lib) init();
    if (nt_enabled) nt_retire(0, context);
    CUresult result = real_cuCtxDestroy_v2(context);
    if (result == CUDA_SUCCESS) { nt_forget_context(context); nt_reclaim(context); }
    else if (nt_enabled) nt_reopen(0, context);
    return result;
}

CUresult cuCtxDestroy(CUcontext context) {
    if (!shared_lib) init();
    if (nt_enabled) nt_retire(0, context);
    CUresult result = real_cuCtxDestroy(context);
    if (result == CUDA_SUCCESS) { nt_forget_context(context); nt_reclaim(context); }
    else if (nt_enabled) nt_reopen(0, context);
    return result;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *context, CUdevice device) {
    if (!shared_lib) init();
    CUresult result = real_cuDevicePrimaryCtxRetain(context, device);
    if (nt_enabled && result == CUDA_SUCCESS) {
        pthread_mutex_lock(&nt_mutex);
        nt_primary **link = &nt_primaries;
        while (*link && !((*link)->device == device && (*link)->context == *context)) link = &(*link)->next;
        nt_primary *p = *link;
        if (p) *link = p->next;
        else p = calloc(1, sizeof(*p));
        if (p) {
            p->device = device; p->context = *context;
            p->next = nt_primaries; nt_primaries = p;
        }
        pthread_mutex_unlock(&nt_mutex);
        if (!p) nt_gap(NULL, "primary-context-tracking-allocation-failed");
    }
    return result;
}

static CUresult nt_primary_end(CUdevice device, bool reset, CUresult (*call)(CUdevice)) {
    nt_primary *primary = NULL;
    CUcontext context = NULL;
    unsigned long long before = 0, after = 0;
    if (nt_enabled) {
        pthread_mutex_lock(&nt_mutex);
        for (nt_primary *p = nt_primaries; p; p = p->next) {
            if (p->device != device) continue;
            primary = p; context = p->context; ++p->retiring;
            break;
        }
        pthread_mutex_unlock(&nt_mutex);
    }
    bool identified = context && real_cuCtxGetId(context, &before) == CUDA_SUCCESS;
    if (context) nt_retire(0, context);
    CUresult result = call(device);
    if (context) {
        unsigned int flags;
        int active = 1;
        /* Observed retains are not the driver's reference count: the runtime or
         * another library may own references that never passed through us. */
        bool inactive = result == CUDA_SUCCESS &&
            real_cuDevicePrimaryCtxGetState(device, &flags, &active) == CUDA_SUCCESS && !active;
        bool replaced = result == CUDA_SUCCESS && identified &&
            real_cuCtxGetId(context, &after) == CUDA_SUCCESS && before != after;
        if (result == CUDA_SUCCESS && (reset || inactive || replaced)) {
            nt_forget_context(context);
            nt_reclaim(context);
        }
        pthread_mutex_lock(&nt_mutex);
        --primary->retiring;
        /* Keep the launch gate closed until every concurrent lifecycle call has
         * finished. No bookkeeping lock is held while draining or calling CUDA. */
        if (!nt_context_closing(context)) {
            for (nt_binding *b = nt_bindings; b; b = b->next)
                if (b->context == context) b->closing = false;
        }
        pthread_mutex_unlock(&nt_mutex);
    }
    return result;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice device) {
    if (!shared_lib) init();
    return nt_primary_end(device, false, real_cuDevicePrimaryCtxRelease);
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    if (!shared_lib) init();
    return nt_primary_end(device, false, real_cuDevicePrimaryCtxRelease_v2);
}

CUresult cuDevicePrimaryCtxReset(CUdevice device) {
    if (!shared_lib) init();
    return nt_primary_end(device, true, real_cuDevicePrimaryCtxReset);
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice device) {
    if (!shared_lib) init();
    return nt_primary_end(device, true, real_cuDevicePrimaryCtxReset_v2);
}

CUresult cuFuncSetCacheConfig(CUfunction function, CUfunc_cache config) {
    if (!shared_lib) init();
    CUresult result = real_cuFuncSetCacheConfig(function, config);
    if (nt_enabled && result == CUDA_SUCCESS) nt_preference_set(NT_FUNCTION, function, nt_current(), -1, false, config);
    return result;
}

CUresult cuFuncSetSharedMemConfig(CUfunction function, CUsharedconfig config) {
    if (!shared_lib) init();
    CUresult result = real_cuFuncSetSharedMemConfig(function, config);
    if (nt_enabled && result == CUDA_SUCCESS) nt_preference_set(NT_FUNCTION, function, nt_current(), -1, true, config);
    return result;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice device) {
    if (!shared_lib) init();
    CUresult result = real_cuKernelSetCacheConfig(kernel, config, device);
    if (result == CUDA_SUCCESS) nt_preference_set(NT_KERNEL, kernel, NULL, device, false, config);
    return result;
}

CUresult cuStreamSynchronize(CUstream stream) {
    if (!shared_lib) init();
    return real_cuStreamSynchronize(stream);
}

CUresult cuStreamSynchronize_ptsz(CUstream stream) {
    if (!shared_lib) init();
    return real_cuStreamSynchronize_ptsz(stream);
}

CUresult cuCtxSynchronize(void) {
    if (!shared_lib) init();
    return real_cuCtxSynchronize();
}

CUresult cuCtxSynchronize_v2(CUcontext context) {
    if (!shared_lib) init();
    return real_cuCtxSynchronize_v2(context);
}

#endif
