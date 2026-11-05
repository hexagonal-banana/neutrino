"""Automatic precompiled/JIT interception fixture; GPU execution is opt-in."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.runtime import (catalog_loop_depth, compile_native, compiler_environment,
                                       image_manifest, implementation_identity, prepare_jit, runtime_environment)
from neutrino.sys_sync.linkage import link_group
from neutrino.sys_sync.probe import instrument_instructions
from neutrino.sys_sync.records import RECORD, STATE, COMMITTED, BufferLayout
from neutrino.sys_sync.sampling import CONTEXT_ABI, policy_json
from neutrino.utils.reconstruct_cuda import sha256

KERNEL = r'''
#include <cuda_runtime.h>
extern "C" {
__device__ unsigned bias = 7;
__device__ unsigned invocations = 0;
struct alignas(16) Payload { unsigned* out; unsigned char padding[4088]; };
__global__ void fixture_kernel(Payload payload) {
    unsigned local = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned thread = block * (blockDim.x * blockDim.y * blockDim.z) + local;
    unsigned* out = payload.out;
    unsigned value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(out + thread) : "memory");
    value += bias;
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(out + thread), "r"(value) : "memory");
    asm volatile("fence.sc.sys;" ::: "memory");
    if (thread == 0) atomicAdd(&invocations, 1);
}
}
'''
HOST = r'''
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <thread>
#define RT(call) do { auto rc = (call); if (rc != cudaSuccess) { std::fprintf(stderr, "%s: %d\n", #call, int(rc)); return 1; } } while(0)
#define DR(call) do { auto rc = (call); if (rc != CUDA_SUCCESS) { std::fprintf(stderr, "%s: %d\n", #call, int(rc)); return 2; } } while(0)
template<class T> T api(void* handle, const char* name) {
    void* ptr = dlsym(handle, name);
    if (!ptr) { std::fprintf(stderr, "Missing %s\n", name); std::exit(3); }
    return reinterpret_cast<T>(ptr);
}
extern "C" int fixture_run(const char* cubin, int reset) {
    void* driver = dlopen(std::getenv("NEUTRINO_REAL_DRIVER"), RTLD_NOW | RTLD_LOCAL);
    if (!driver) return 4;
    auto begin = api<int(*)(const char*, std::int64_t)>(RTLD_DEFAULT, "neutrinoSysCaptureBegin");
    auto end = api<int(*)(int)>(RTLD_DEFAULT, "neutrinoSysCaptureEnd");
    auto finish = api<int(*)()>(RTLD_DEFAULT, "neutrinoSysFinalize");
    RT(cudaSetDevice(0));
    unsigned* out = nullptr;
    RT(cudaMalloc(&out, 32 * sizeof(unsigned)));
    RT(cudaMemset(out, 0, 32 * sizeof(unsigned)));
    unsigned changed = 11;
    RT(cudaMemcpyToSymbol(bias, &changed, sizeof(changed)));
    CUfunction registered{};
    RT(cudaGetFuncBySymbol(&registered, reinterpret_cast<const void*>(fixture_kernel)));
    void* queried = nullptr;
    cudaDriverEntryPointQueryResult query{};
    RT(cudaGetDriverEntryPointByVersion("cuLaunchKernelEx", &queried, 12000, cudaEnableDefault, &query));
    if (!queried || query != cudaDriverEntryPointSuccess) return 5;
    auto extended = reinterpret_cast<decltype(&cuLaunchKernelEx)>(queried);
    auto load = api<decltype(&cuLibraryLoadFromFile)>(driver, "cuLibraryLoadFromFile");
    auto enumerate = api<decltype(&cuLibraryEnumerateKernels)>(driver, "cuLibraryEnumerateKernels");
    auto get_function = api<decltype(&cuKernelGetFunction)>(driver, "cuKernelGetFunction");
    auto unload = api<decltype(&cuLibraryUnload)>(driver, "cuLibraryUnload");
    CUlibrary library{}; CUkernel kernel{}; CUfunction jit{};
    DR(load(&library, cubin, nullptr, nullptr, 0, nullptr, nullptr, 0));
    DR(enumerate(&kernel, 1, library));
    DR(get_function(&jit, kernel));
    Payload payload{}; payload.out = out;
    void* args[]{&payload};
    CUlaunchAttribute attribute{};
    attribute.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    attribute.value.cooperative = 1;
    CUlaunchConfig config{1,1,1,32,1,1,0,nullptr,&attribute,1};
    std::size_t size = 16;
    void* packed[]{CU_LAUNCH_PARAM_BUFFER_POINTER, &payload, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
    cudaStream_t worker_stream{};
    RT(cudaStreamCreateWithFlags(&worker_stream, cudaStreamNonBlocking));
    if (begin("automatic-interception", 1)) return 6;
    RT(cudaLaunchKernel(reinterpret_cast<const void*>(fixture_kernel), dim3(1), dim3(32), args, 0, nullptr));
    DR(extended(&config, registered, nullptr, packed));
    cudaLaunchConfig_t runtime_config{};
    runtime_config.gridDim = dim3(1); runtime_config.blockDim = dim3(32);
    RT(cudaLaunchKernelExC(&runtime_config, reinterpret_cast<const void*>(fixture_other_kernel), args));
    DR(extended(&config, jit, args, nullptr));
    RT(cudaDeviceSynchronize());
    CUlaunchConfig threaded = config; threaded.hStream = worker_stream;
    int worker_status = -1;
    std::thread worker([&] {
        CUcontext context{};
        auto current = api<decltype(&cuCtxGetCurrent)>(driver, "cuCtxGetCurrent");
        if (current(&context) != CUDA_SUCCESS || context) return;
        worker_status = int(extended(&threaded, registered, args, nullptr));
        if (current(&context) != CUDA_SUCCESS || context) worker_status = -2;
    });
    worker.join();
    if (worker_status) return 12;
    RT(cudaDeviceSynchronize());
    if (end(1)) return 7;
    unsigned values[32]{};
    RT(cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost));
    for (auto value : values) if (value != 51) return 8;
    unsigned calls = 0;
    RT(cudaMemcpyFromSymbol(&calls, invocations, sizeof(calls)));
    if (calls != 4) return 9;
    auto global = api<decltype(&cuLibraryGetGlobal)>(driver, "cuLibraryGetGlobal");
    auto copy = api<decltype(&cuMemcpyDtoH_v2)>(driver, "cuMemcpyDtoH_v2");
    CUdeviceptr address{}; std::size_t bytes = 0;
    DR(global(&address, &bytes, library, "invocations"));
    DR(copy(&calls, address, sizeof(calls)));
    if (calls != 1) return 10;
    DR(unload(library));
    RT(cudaFree(out));
    RT(cudaStreamDestroy(worker_stream));
    if (finish()) return 11;
    if (reset) RT(cudaDeviceReset());
    return 0;
}
'''


def compile_fixture(directory, *, jit_iterations=64, interval_mode='instruction-only', loop_depth=1,
                    jit_loop_depth=None, grid=(1, 1, 1), block=(32, 1, 1)):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
    env = compiler_environment()
    source = directory / 'fixture.cu'
    from math import prod
    threads = prod(grid) * prod(block)
    alias = KERNEL[KERNEL.index('__global__'):KERNEL.rfind('}')].replace('fixture_kernel', 'fixture_other_kernel')
    host = HOST.replace('32 * sizeof(unsigned)', f'{threads} * sizeof(unsigned)')
    host = host.replace('unsigned values[32]', f'unsigned values[{threads}]')
    host = host.replace('dim3(1)', 'dim3(' + ','.join(map(str, grid)) + ')')
    host = host.replace('dim3(32)', 'dim3(' + ','.join(map(str, block)) + ')')
    host = host.replace('config{1,1,1,32,1,1,', 'config{' + ','.join(map(str, (*grid, *block))) + ',')
    source.write_text(KERNEL + '\nextern "C" {\n' + alias + '\n}\n' + host)
    shared = directory / 'fixture.so'
    command = [str(toolchain / 'bin/nvcc'), '-arch=sm_90', '-O3', '-lineinfo',
               '--keep', '--keep-dir', str(directory), '--cudart=shared', '-shared',
               '-Xcompiler=-fPIC', str(source), '-ldl', '-o', str(shared)]
    with (directory / 'compile.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    ptx = directory / 'fixture.ptx'
    original = directory / 'fixture.sm_90.cubin'
    fatbin = directory / 'fixture.fatbin'
    transformed = instrument_instructions(link_group({'module': ptx.read_text()}),
                                           max_parameter_bytes=32764, interval_mode=interval_mode, loop_depth=loop_depth)
    probed = directory / 'probed.ptx'
    probed.write_text(transformed.sources['module'])
    replacement = directory / 'probed.cubin'
    subprocess.run([str(toolchain / 'bin/ptxas'), '-arch=sm_90', '-O3', '-lineinfo',
                    str(probed), '-o', str(replacement)], env=env, check=True, capture_output=True, timeout=60)
    packaged = directory / 'probed.fatbin'
    subprocess.run([str(toolchain / 'bin/fatbinary'), '--64', '--create=' + str(packaged),
                    '--image3=kind=elf,sm=90,file=' + str(replacement)], env=env, check=True,
                   capture_output=True, timeout=60)
    precompiled = image_manifest(original, packaged, transformed.manifest, 'precompiled-fixture', aliases=(fatbin,))
    precompiled['target'] = 'sm_90'
    precompiled['entries']['fixture_kernel']['packed_prefix_min'] = 16
    jit = directory / 'jit'
    jit.mkdir(exist_ok=True)
    # One hot thread exceeds the configured 16-record average without losing
    # any observations; a fixed per-thread allocation would overflow here.
    skewed = KERNEL.replace('    if (thread == 0) atomicAdd',
                            f'    for (unsigned i = 0; thread == 0 && i < {jit_iterations}; ++i) '
                            'asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(out) : "memory");\n'
                            '    if (thread == 0) atomicAdd')
    (jit / 'kernel.cu').write_text(skewed)
    for mode, suffix in (('--cubin', '.cubin'), ('--ptx', '.ptx')):
        subprocess.run([str(toolchain / 'bin/nvcc'), '-arch=sm_90a', '-O3', '-lineinfo',
                        mode, str(jit / 'kernel.cu'), '-o', str(jit / ('kernel' + suffix))],
                       env=env, check=True, capture_output=True, timeout=60)
    (jit / 'meta.json').write_text(json.dumps({'compiler_options': {
        'arch': '90a', 'optimize_level': '3', 'ptxas_register_usage_level': 10, 'with_line_info': True}}))
    catalog = directory / 'catalog.json'
    catalog.write_text(json.dumps({'schema': 1, 'context_abi': CONTEXT_ABI, 'interval_mode': interval_mode, 'loop_depth': loop_depth,
                                   'jit_loop_depth': loop_depth if jit_loop_depth is None else jit_loop_depth,
                                   'images': [precompiled], 'jit_roots': [str(jit)]}, indent=2))
    report = {'shared': str(shared), 'catalog': str(catalog), 'jit': str(jit / 'kernel.cubin'),
              'sites': len(transformed.manifest['sites']), 'compile_command': command}
    (directory / 'fixture.json').write_text(json.dumps(report, indent=2))
    return report


def validate_records(output, interval_mode='instruction-only', loop_depth=1, jit_loop_depth=None, exclude_image=None):
    processes = list(Path(output).glob('process-*'))
    if len(processes) != 1:
        raise AssertionError('Expected exactly one native process journal')
    directory = processes[0]
    journal = [json.loads(line) for line in (directory / 'journal.jsonl').read_text().splitlines()]
    if any(row['kind'] == 'fatal' for row in journal):
        raise AssertionError('Native fatal event')
    launches = [row for row in journal if row['kind'] == 'launch' and row['step'] == 1]
    assert len(launches) == 5 and all(row['status'] == 0 for row in launches)
    excluded_label = {'jit': 'DeepEP', 'precompiled': 'precompiled-fixture'}.get(exclude_image)
    for row in launches:
        image = json.loads((directory / (row['image'] + '.image.json')).read_text())
        excluded = image['label'] == excluded_label
        assert row['coverage'] == ('observed' if excluded else 'instrumented')
        assert image.get('probe_enabled', True) is not excluded
        assert ('raw' in row) is not excluded
    assert len({row['owner'] for row in launches}) == 2
    assert [row['api'] for row in launches] == ['cudaLaunchKernel', 'cuLaunchKernelEx', 'cudaLaunchKernelExC',
                                               'cuLaunchKernelEx', 'cuLaunchKernelEx']
    intervals = 0
    for launch in launches:
        if launch['coverage'] == 'observed':
            continue
        data = (directory / launch['raw']).read_bytes()
        threads, capacity = launch['threads'], launch['capacity']
        packed = launch.get('raw_format') in {'states-packed-v1', 'states-packed-v2'}
        image = json.loads((directory / (launch['image'] + '.image.json')).read_text())
        sites = {row['ordinal'] for row in image['probe']['sites']}
        pooled = launch.get('record_storage') == 'pooled-v1'
        assert image['interval_mode'] == interval_mode
        expected_depth = jit_loop_depth if image['label'] == 'DeepEP' and jit_loop_depth is not None else loop_depth
        assert image.get('loop_depth', 1) == image['probe'].get('loop_depth', 1) == expected_depth
        extra = (1 if interval_mode == 'loop-boundary' else 64) if pooled else 0
        layout = BufferLayout.from_dimensions(launch['grid'], launch['block'], capacity, launch['capture_sampling'])
        assert threads == layout.threads and len(data) == (threads * STATE.size + (threads * 2 + extra) * RECORD.size
                                               if packed else threads * (STATE.size + capacity * RECORD.size))
        cursor = threads * STATE.size
        for thread in range(threads):
            used = 2 + (extra if thread == 0 else 0)
            assert STATE.unpack_from(data, thread * STATE.size) == (used, 0, 1, 0)
            previous = 0
            for ordinal in range(used if packed else capacity):
                offset = cursor if packed else threads * STATE.size + (thread * capacity + ordinal) * RECORD.size
                record = RECORD.unpack_from(data, offset)
                if packed:
                    cursor += RECORD.size
                if ordinal >= used:
                    assert record == (0, 0, 0, 0, 0)
                    continue
                begin, end, site, recorded, commit = record
                assert 0 < begin <= end and begin >= previous and recorded == ordinal and commit == COMMITTED and site in sites
                previous = end
                intervals += 1
    assert any(row['kind'] == 'capture_end' and row['successful'] for row in journal)
    assert any(row['kind'] == 'finalized' for row in journal)
    calibrations = [row for row in journal if row['kind'] == 'calibration']
    assert {row['phase'] for row in calibrations} == {'before', 'after'}
    assert all(len(row['samples']) == 32 and all(0 < a <= b and tick > 0 for a, tick, b in row['samples']) for row in calibrations)
    from neutrino.sys_sync.export import NativeProcess
    process = NativeProcess(directory / 'journal.jsonl')
    for row in process.launches:
        process.decode_launch(row)
    return {'launches': len(launches), 'intervals': intervals,
            'excluded_launches': sum(row['coverage'] == 'observed' for row in launches),
            'excluded_image': exclude_image,
            'all_threads_complete': exclude_image is None and all(row['capture_sampling'] == 'all' for row in launches),
            'selected_threads_complete': True,
            'runtime_globals_preserved': True, 'exactly_once': True, 'jit_resolver_executed': exclude_image is None,
            'skewed_pool_thread_records': 3 if interval_mode == 'loop-boundary' else 66,
            'interval_mode': interval_mode, 'loop_depth': loop_depth,
            'jit_loop_depth': loop_depth if jit_loop_depth is None else jit_loop_depth}


def validate_observation(output, mode, steps=(1,), ranks=(0,)):
    """Validate original-image observation without claiming a GPU-detail trace."""
    from collections import Counter
    assert mode in ('hooks', 'hooks-cupti')
    summaries = []
    for path in sorted(Path(output).glob('process-*/journal.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        identities = [r for r in rows if r['kind'] == 'process']
        assert len(identities) == 1
        process = identities[0]
        assert process['capture_mode'] == mode and process['kernel_probes'] is False
        assert process['cupti_enabled'] == (mode == 'hooks-cupti')
        assert not any(r['kind'] in ('fatal', 'calibration', 'drain') for r in rows)
        assert not list(path.parent.glob('launch-*.bin'))
        if not any(r['kind'] == 'capture_begin' for r in rows):
            # Launcher/helper processes can inherit the interceptor without capturing.
            assert not any(r['kind'] in ('capture_end', 'kernel_activity', 'activity_validation') or
                           'raw' in r or (r['kind'] == 'launch' and r['step'] != -1) for r in rows)
            continue
        assert any(r['kind'] == 'finalized' for r in rows)
        assert all(a['host_ns'] <= b['host_ns'] for a, b in zip(rows, rows[1:]))
        begins = [r for r in rows if r['kind'] == 'capture_begin']
        ends = [r for r in rows if r['kind'] == 'capture_end']
        assert [r['step'] for r in begins] == list(steps) == [r['step'] for r in ends]
        assert all(r['successful'] and r['recorded_launches'] == 0 for r in ends)
        launches = [r for r in rows if r['kind'] == 'launch']
        assert len({r['ordinal'] for r in launches}) == len(launches)
        assert all(r['status'] == 0 and 'raw' not in r and
                   r['coverage'] in ('observed', 'pass_through') for r in launches)
        selected = [r for r in launches if r['coverage'] == 'observed' and r['step'] in steps]
        owners = {r['owner']: r for r in rows if r['kind'] == 'load'}
        contexts = {r['context']: r for r in rows if r['kind'] == 'context'}
        images = {}
        for row in selected:
            owner = owners[row['owner']]
            assert owner['coverage'] == 'original' and owner['image'] == row['image']
            assert contexts[row['context']]['uuid'] == row['uuid']
            if row['image'] not in images:
                image = json.loads((path.parent / (row['image'] + '.image.json')).read_text())
                assert sha256(image['original']) == image['id'] == row['image']
                images[row['image']] = image
            assert row['kernel'] in images[row['image']]['entries']
        activity = [r for r in rows if r['kind'] == 'kernel_activity']
        checks = [r for r in rows if r['kind'] == 'activity_validation']
        for begin, end in zip(begins, ends):
            expected = {r['ordinal']: r for r in selected if r['step'] == begin['step']}
            assert expected and len(expected) == end['observed_launches']
            assert all(begin['host_ns'] <= r['host_ns'] <= end['host_ns'] for r in expected.values())
        if mode == 'hooks-cupti':
            assert len(activity) == len(selected)
            assert {r['ordinal'] for r in activity} == {r['ordinal'] for r in selected}
            assert [r['step'] for r in checks] == list(steps)
            by_ordinal = {r['ordinal']: r for r in selected}
            windows = {a['step']: (a['host_ns'], b['host_ns']) for a, b in zip(begins, ends)}
            for row in activity:
                launch = by_ordinal[row['ordinal']]
                assert row['step'] == launch['step']
                assert row['grid'] == launch['grid'] and row['block'] == launch['block']
                assert row['kernel'] == launch['kernel']
                assert row['device'] == contexts[launch['context']]['device']
                assert row['source'] == 'CUPTI_CONCURRENT_KERNEL' and row['host_clock'] == 'CLOCK_MONOTONIC_RAW'
                low, high = windows[row['step']]
                assert low <= row['begin_host_ns'] < row['end_host_ns'] <= high
            assert all(r['dropped'] == 0 and r['matched'] == sum(a['step'] == r['step'] for a in activity)
                       for r in checks)
        else:
            assert not activity and not checks
        summaries.append({'rank': int(process['rank']), 'pid': process['pid'], 'session': process['session'],
                          'selected_launches': len(selected), 'kernel_spans': len(activity),
                          'all_launches': len(launches), 'measured_launches': sum(r['step'] in steps for r in launches),
                          'library_counts': dict(Counter(images[r['image']]['label'] for r in selected)),
                          'step_kernel_counts': {str(step): dict(Counter(
                              images[r['image']]['label'] for r in selected if r['step'] == step)) for step in steps},
                          'original_image_hashes': sorted(images), 'journal_bytes': path.stat().st_size})
    assert sorted(r['rank'] for r in summaries) == list(ranks)
    return {'status': 'passed', 'capture_mode': mode, 'kernel_probes': False,
            'raw_records': 0, 'cupti_dropped_records': 0 if mode == 'hooks-cupti' else None,
            'ranks': summaries}


class RuntimeTests(unittest.TestCase):
    def test_independent_jit_depth_and_catalog_validation(self):
        from neutrino.sys_sync.ptx import PTXError, SITE_SELECTION
        for mode, default in (('instruction-only', 1), ('loop-boundary', 3)):
            catalog = {'interval_mode': mode, 'loop_depth': default}
            self.assertEqual(catalog_loop_depth(catalog, 'DeepEP'), default)
            for invalid in (0, -1, True, '4', 1.5, None, 2147483648):
                with self.subTest(invalid=invalid), self.assertRaises(PTXError):
                    catalog_loop_depth(dict(catalog, jit_loop_depth=invalid), 'NCCL')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replacement = root / 'replacement'
            replacement.write_bytes(b'fixture')
            build = root / 'build.json'
            build.write_text(json.dumps({'status': 'built', 'implementation': implementation_identity(),
                                         'library': str(replacement), 'sha256': sha256(replacement)}))
            catalog = {'schema': 1, 'context_abi': CONTEXT_ABI, 'interval_mode': 'loop-boundary',
                       'loop_depth': 3, 'jit_loop_depth': 4, 'images': []}
            for library, depth in (('NCCL', 3), ('DeepEP', 4)):
                self.assertEqual(catalog_loop_depth(catalog, library), depth)
                catalog['images'].append({'label': library, 'context_abi': CONTEXT_ABI,
                    'interval_mode': 'loop-boundary', 'loop_depth': depth,
                    'replacement': str(replacement), 'replacement_sha256': sha256(replacement),
                    'probe': {'context_abi': CONTEXT_ABI, 'site_selection': SITE_SELECTION,
                              'sampling': 'launch_policy', 'loop_depth': depth}})
            path = root / 'catalog.json'
            path.write_text(json.dumps(catalog))
            runtime_environment(build, path, root / 'raw', 'fixture')
            for enabled in (True, False, 0, 1, 'false', None, []):
                catalog['images'][1]['probe_enabled'] = enabled
                path.write_text(json.dumps(catalog))
                if type(enabled) is bool:
                    runtime_environment(build, path, root / 'raw', 'fixture')
                else:
                    with self.assertRaisesRegex(PTXError, 'probe_enabled'):
                        runtime_environment(build, path, root / 'raw', 'fixture')
            catalog['images'][1].pop('probe_enabled')
            for depth in (3, True):
                catalog['images'][1]['loop_depth'] = depth
                path.write_text(json.dumps(catalog))
                with self.assertRaises(PTXError):
                    runtime_environment(build, path, root / 'raw', 'fixture')
            with self.assertRaises(PTXError):
                catalog_loop_depth({'interval_mode': 'instruction-only', 'jit_loop_depth': 4}, 'DeepEP')

    def test_capture_mode_validation(self):
        from neutrino.sys_sync.ptx import PTXError
        for mode in ('', 'FULL', 'cupti', None, False, []):
            with self.subTest(mode=mode), self.assertRaisesRegex(PTXError, 'capture mode'):
                runtime_environment('absent', 'absent', 'absent', 'test', capture_mode=mode)

    def test_observation_ignores_only_noncollecting_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'process-1/journal.jsonl'
            path.parent.mkdir()
            process = {'kind': 'process', 'capture_mode': 'hooks', 'kernel_probes': False,
                       'cupti_enabled': False}
            path.write_text(json.dumps(process) + '\n')
            self.assertEqual(validate_observation(temporary, 'hooks', ranks=())['ranks'], [])
            for extra in ({'kind': 'fatal'}, {'kind': 'capture_end'}, {'kind': 'kernel_activity'},
                          {'kind': 'launch', 'step': 3}):
                path.write_text(json.dumps(process) + '\n' + json.dumps(extra) + '\n')
                with self.subTest(extra=extra), self.assertRaises(AssertionError):
                    validate_observation(temporary, 'hooks', ranks=())

    def test_observation_export_rejected(self):
        from neutrino.sys_sync.export import NativeProcess, TraceError
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'journal.jsonl'
            for mode in ('hooks', 'hooks-cupti', 'invalid'):
                path.write_text(json.dumps({'kind': 'process', 'capture_mode': mode}) + '\n')
                with self.subTest(mode=mode), self.assertRaisesRegex(TraceError, 'Observation-only'):
                    NativeProcess(path)

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected toolchain required')
    def test_precompiled_fixture_and_jit_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = compile_fixture(root)
            self.assertEqual(report['sites'], 4)
            resolved = prepare_jit(report['jit'], root / 'cache')
            image = json.loads(resolved.read_text())
            self.assertTrue(image['sass_equal'] and image['resources_equal'])
            self.assertEqual(image['probe']['record_storage'], 'pooled-v1')
            self.assertEqual(len(image['probe']['sites']), 3)
            self.assertEqual(prepare_jit(report['jit'], root / 'cache'), resolved)
            loop_path = prepare_jit(report['jit'], root / 'cache', interval_mode='loop-boundary')
            loop_image = json.loads(loop_path.read_text())
            self.assertNotEqual(loop_path, resolved)
            self.assertEqual(loop_image['interval_mode'], 'loop-boundary')
            self.assertEqual(len(loop_image['probe']['loops']), 1)
            self.assertEqual(sum(s['interval_kind'] == 'loop' for s in loop_image['probe']['sites']), 1)
            depth_path = prepare_jit(report['jit'], root / 'cache', interval_mode='loop-boundary', loop_depth=2)
            self.assertNotEqual(depth_path, loop_path)
            depth_image = json.loads(depth_path.read_text())
            self.assertEqual(depth_image['loop_depth'], 2)
            self.assertEqual(depth_image['probe']['loop_depth'], 2)
            self.assertEqual(depth_image['probe']['loops'][0]['contained_sites'][0]['selected_loop_depth'], 1)
            self.assertEqual(prepare_jit(report['jit'], root / 'cache', interval_mode='loop-boundary', loop_depth=2),
                             depth_path)

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected toolchain required')
    def test_runtime_exports_and_disabled_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = compile_native(Path(temporary) / 'build')
            env = compiler_environment()
            env['LD_PRELOAD'] = report['library']
            code = 'import ctypes; assert ctypes.CDLL(None).neutrinoSysAbiVersion() == 0'
            subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=30)
            self.assertEqual(sha256(report['library']), report['sha256'])
            catalog = Path(temporary) / 'catalog.json'
            catalog.write_text(json.dumps({'schema': 1, 'context_abi': CONTEXT_ABI, 'interval_mode': 'instruction-only', 'images': []}))
            settings = runtime_environment(Path(temporary) / 'build/build.json', catalog,
                                           Path(temporary) / 'raw', 'cpu-test')
            self.assertEqual(settings['NEUTRINO_SYS_CAPTURE_MODE'], 'full')
            for mode in ('hooks', 'hooks-cupti'):
                observed = runtime_environment(Path(temporary) / 'build/build.json', catalog,
                                               Path(temporary) / mode, 'cpu-test', capture_mode=mode)
                self.assertEqual(observed['NEUTRINO_SYS_CAPTURE_MODE'], mode)
            from neutrino.sys_sync.ptx import PTXError
            stale = json.loads(catalog.read_text())
            for selection in (None, 'explicit_sys_v1', 'acquire_sys_and_all_fences_barriers_v1',
                              'acquire_sys_system_fences_all_mbarrier_v1'):
                stale['images'] = [{'probe': {'site_selection': selection}}]
                catalog.write_text(json.dumps(stale))
                with self.assertRaisesRegex(PTXError, 'instruction coverage'):
                    runtime_environment(Path(temporary) / 'build/build.json', catalog,
                                        Path(temporary) / 'stale', 'cpu-test')
            child = {**compiler_environment(), 'PYTHONPATH': settings['PYTHONPATH']}
            child.pop('LD_PRELOAD', None)
            subprocess.run([sys.executable, '-c', 'import neutrino.sys_sync.runtime'], cwd=temporary,
                           env=child, check=True, timeout=30)


def inspect_capture(root):
    """Summarize native thread-state failures without accepting a partial trace."""
    import collections
    result = []
    for journal in sorted(Path(root).glob('process-*/journal.jsonl')):
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row['kind'] != 'launch' or 'raw' not in row:
                continue
            raw = journal.parent / row['raw']
            item = {'process': journal.parent.name, 'launch': row['ordinal'], 'kernel': row['kernel'],
                    'threads': row['threads'], 'capacity': row['capacity']}
            if not raw.exists():
                item['raw'] = 'missing'
            else:
                with raw.open('rb') as stream:
                    data = stream.read(row['threads'] * STATE.size)
                states = list(STATE.iter_unpack(data))
                item.update(max_used=max((s[0] for s in states), default=0),
                            overflow_threads=sum(s[1] != 0 for s in states),
                            incomplete_threads=sum(s[2] != 1 for s in states),
                            states=dict(collections.Counter(str(s) for s in states)))
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allocation-confirmed', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--inspect', type=Path)
    parser.add_argument('--interval-mode', choices=('instruction-only', 'loop-boundary'), default='instruction-only')
    parser.add_argument('--loop-depth', type=int, default=1)
    parser.add_argument('--jit-loop-depth', type=int)
    parser.add_argument('--sampling', choices=('all', 'warp', 'block'), default='all')
    parser.add_argument('--capture-mode', choices=('full', 'hooks', 'hooks-cupti'), default='full')
    parser.add_argument('--multidimensional', action='store_true')
    parser.add_argument('--exclude-image', choices=('jit', 'precompiled'),
                        help='Execute one whole image unmodified during a full mixed capture')
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_capture(args.inspect), indent=2))
        return
    if args.worker:
        report = json.loads(args.worker.read_text())
        shared = ctypes.CDLL(report['shared'], mode=ctypes.RTLD_GLOBAL)
        shared.fixture_run.argtypes = [ctypes.c_char_p, ctypes.c_int]
        rc = shared.fixture_run(report['jit'].encode(), 1)
        if rc:
            raise RuntimeError(f'Automatic interception numerical fixture failed: {rc}')
        return
    if not args.execute:
        print(json.dumps({'gpu_execution': False}))
        return
    if not args.allocation_confirmed or args.output is None:
        parser.error('--execute requires allocation confirmation and a fresh output')
    if args.exclude_image and args.capture_mode != 'full':
        parser.error('--exclude-image requires full capture mode')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    build = compile_native(root / 'native')
    fixture = compile_fixture(root / 'fixture', interval_mode=args.interval_mode, loop_depth=args.loop_depth,
                              jit_loop_depth=args.jit_loop_depth,
                              **({'grid': (2, 2, 1), 'block': (7, 5, 1)} if args.multidimensional else {}))
    if args.capture_mode == 'full':
        overflow_fixture = compile_fixture(root / 'overflow-fixture', jit_iterations=4096)
    if args.capture_mode != 'full' or args.exclude_image:
        image = prepare_jit(fixture['jit'], root / 'prepared-jit', interval_mode=args.interval_mode,
                            loop_depth=args.loop_depth if args.jit_loop_depth is None else args.jit_loop_depth)
        catalog = json.loads(Path(fixture['catalog']).read_text())
        catalog['images'].append(json.loads(image.read_text()))
        if args.exclude_image:
            catalog['images'][1 if args.exclude_image == 'jit' else 0]['probe_enabled'] = False
        Path(fixture['catalog']).write_text(json.dumps(catalog, indent=2) + '\n')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/moe_training'))
    from gpu_guard import launch_lock, require_idle
    with launch_lock():
        devices = require_idle()
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': devices[0]['uuid'],
               'PYTHONPATH': str(Path(__file__).resolve().parents[1]),
               'LD_PRELOAD': build['library'] + ':' + os.environ.get('LD_PRELOAD', ''),
               'NEUTRINO_SESSION_ID': str(uuid.uuid4()), 'NEUTRINO_SYS_CATALOG': fixture['catalog'],
               'NEUTRINO_SYS_OUTPUT': str(root / 'raw'), 'NEUTRINO_SYS_CAPACITY': '16',
               'NEUTRINO_SYS_BUDGET': str(64 * 1024**2), 'NEUTRINO_SYS_PYTHON': sys.executable,
               'NEUTRINO_SYS_CACHE': str(root / 'jit-cache'), 'NEUTRINO_SYS_CAPTURE_MODE': args.capture_mode,
               'NEUTRINO_SYS_CAPTURE_POLICY': policy_json({'schema': 1, 'default': args.sampling,
                    'rules': [{'kernel_glob': 'fixture_other_kernel', 'library': 'precompiled-fixture', 'sampling': 'all'}]})}
        with (root / 'run.log').open('w') as log:
            result = subprocess.run([sys.executable, __file__, '--worker', str(root / 'fixture/fixture.json')],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        report = {'returncode': result.returncode, 'devices': devices, 'build': build}
        if result.returncode == 0 and args.capture_mode != 'full':
            report.update(validate_observation(root / 'raw', args.capture_mode))
            assert report['ranks'][0]['selected_launches'] == 5
            assert not (root / 'jit-cache').exists(), 'Observer unexpectedly invoked the JIT resolver'
            report.update(runtime_globals_preserved=True, exactly_once=True)
            (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'status': report['status'], 'capture_mode': args.capture_mode}))
            return
        if result.returncode == 0:
            report.update(validate_records(root / 'raw', args.interval_mode, args.loop_depth,
                                           args.jit_loop_depth, args.exclude_image), status='passed')
            if args.exclude_image:
                from neutrino.sys_sync.export import export_trace
                report['export'] = export_trace(root / 'raw', root / 'decoded',
                    require_kernel_activity=True, instruction_granularity='captured')
                assert report['excluded_launches'] == (1 if args.exclude_image == 'jit' else 4)
                if args.interval_mode == 'loop-boundary':
                    from test_sys_loops import validate_loop_export
                    report['export_validation'] = validate_loop_export(root / 'decoded')
            # NVML utilization can still cover our just-finished kernel window.
            # Wait for that window to age out, then apply the unchanged guard.
            time.sleep(5)
            require_idle()
            overflow_env = {**env, 'NEUTRINO_SYS_CAPTURE_POLICY': policy_json({'schema': 1, 'default': 'all'}),
                            'NEUTRINO_SYS_CATALOG': overflow_fixture['catalog'],
                            'NEUTRINO_SYS_OUTPUT': str(root / 'expected-overflow')}
            with (root / 'overflow.log').open('w') as log:
                rejected = subprocess.run([sys.executable, __file__, '--worker',
                                           str(root / 'overflow-fixture/fixture.json')],
                                          env=overflow_env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            assert rejected.returncode == 86
            journals = list((root / 'expected-overflow').glob('process-*/journal.jsonl'))
            assert len(journals) == 1
            rows = [json.loads(line) for line in journals[0].read_text().splitlines()]
            pooled = next(row for row in rows if row.get('record_storage') == 'pooled-v1')
            drain = next(row for row in rows if row['kind'] == 'drain' and row['ordinal'] == pooled['ordinal'])
            assert drain['records'] == pooled['capacity'] == 512 and drain['overflow_threads'] > 0
            assert drain['incomplete_threads'] == 0
            assert any(row['kind'] == 'capture_end' and not row['successful'] for row in rows)
            report['pool_overflow_rejected_without_wrap'] = True
        else:
            report['status'] = 'failed'
        (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'Fixture failed without retry; see {root / "run.log"}')
        print(json.dumps({key: report[key] for key in ('status', 'launches', 'intervals')}))


if __name__ == '__main__':
    if any(arg.startswith('--') for arg in sys.argv[1:]):
        main()
    else:
        unittest.main()
