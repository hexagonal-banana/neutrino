"""Diagnostic timestamp emission retains original code and all-thread semantics."""
import argparse
import ctypes as C
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neutrino.sys_sync.linkage import link_group
from neutrino.sys_sync.probe import instrument_instructions
from neutrino.sys_sync.ptx import PTXError, parse
from neutrino.sys_sync.identity import ArtifactIdentity, ProcessIdentity, Registry
from test_sys_native import compile_bridge
from neutrino.sys_sync.records import (BufferLayout, CONTEXT, HEADER as RAW_HEADER, SCHEMA, STATE,
                                       RecordError, decode, encode_header)

HEADER = '.version 9.1\n.target sm_90\n.address_size 64\n'


def instrument(source, mode='instruction-only'):
    return instrument_instructions(link_group({'module': source}), max_parameter_bytes=32764, interval_mode=mode)


class ProbeTests(unittest.TestCase):
    def test_mbarrier_exclusion_preserves_code_and_removes_barrier_only_loops(self):
        from neutrino.sys_sync.ptx import SITE_SELECTION_NO_MBARRIER
        source = HEADER + '''.entry k() {
 .reg .pred %p; .reg .b32 %addr; .reg .b64 %state;
 L: mbarrier.try_wait.shared.b64 %p, [%addr], %state; @!%p bra L;
 membar.sys; ret;
}'''
        group = link_group({'module': source})
        for mode in ('instruction-only', 'loop-boundary'):
            settings = dict(max_parameter_bytes=32764, interval_mode=mode)
            default = instrument_instructions(group, **settings)
            self.assertEqual(default, instrument_instructions(group, mbarrier='include', **settings))
            result = instrument_instructions(group, mbarrier='exclude', **settings)
            self.assertEqual(result.manifest['site_selection'], SITE_SELECTION_NO_MBARRIER)
            self.assertEqual([s['opcode'] for s in result.manifest['sites']], ['membar.sys'])
            self.assertFalse(result.manifest.get('loops'))
            self.assertEqual(group.modules['module'].sites, parse(source).sites)
            for function in parse(source).functions:
                for instruction in function.instructions:
                    self.assertIn(source[instruction.start:instruction.end], result.sources['module'])
            self.assertEqual(instrument_instructions(group, **settings), default)
        with self.assertRaisesRegex(PTXError, 'mbarrier'):
            instrument_instructions(group, max_parameter_bytes=32764,
                                    interval_mode='instruction-only', mbarrier='invalid')

    def test_mbarrier_only_image_retains_entry_completion_without_records(self):
        source = HEADER + '.entry k() { .reg .pred %p; .reg .b32 %a; ' \
            'mbarrier.try_wait.parity.shared.b64 %p, [%a], 0; ret; }'
        result = instrument_instructions(link_group({'module': source}), max_parameter_bytes=32764,
                                         interval_mode='loop-boundary', mbarrier='exclude')
        self.assertEqual(result.manifest['sites'], [])
        self.assertEqual(result.manifest['loops'], [])
        self.assertIn('state+8], 1;', result.sources['module'])
        self.assertNotIn('%globaltimer', result.sources['module'])

    def test_original_system_instructions_and_predicates_remain_exact(self):
        source = HEADER + '''
.entry k(.param .u64 ptr) {
 .reg .u64 %rd; .reg .u32 %r; .reg .pred %p;
 ld.param.u64 %rd, [ptr]; setp.eq.u32 %p, 0, 0;
 LAB: @!%p ld.acquire.sys.global.u32 %r, [%rd];
 @%p st.release.sys.global.u32 [%rd], %r;
 @!%p ret; ret;
}
'''
        result = instrument(source)
        original, changed = parse(source), parse(result.sources['module'])
        self.assertEqual([(s.opcode, s.predicate) for s in original.sites],
                         [(s.opcode, s.predicate) for s in changed.sites])
        for site in original.sites:
            self.assertEqual(result.sources['module'].count(source[site.start:site.end]), 1)
        self.assertEqual(result.manifest['sampling'], 'launch_policy')
        self.assertEqual(result.manifest['sites'][0]['start'], original.sites[0].start)
        self.assertFalse(result.manifest['sass_timer_placement_validated'])
        self.assertFalse(result.manifest['polling_intervals_validated'])
        self.assertFalse(result.manifest['runtime_verified'])
        self.assertNotIn('laneid', result.sources['module'])
        self.assertEqual(result.sources['module'].count('state+8], 1;'), 2)

    def test_loop_keeps_dynamic_instruction_mode_not_invented_polling(self):
        source = HEADER + '.entry k() { .reg .pred %p; L: membar.sys; @%p bra L; ret; }'
        result = instrument(source)
        self.assertEqual(len(result.manifest['sites']), 1)
        self.assertIn('@%p bra L;', result.sources['module'])
        with self.assertRaisesRegex(PTXError, 'instruction-only'):
            instrument(source, mode='polling')

    def test_site_identity_exact_and_independent_of_transformed_offsets(self):
        source = HEADER + '.entry one() { membar.sys; ret; } .entry two() { membar.sys; ret; }'
        first = instrument(source).manifest['sites']
        second = instrument(source).manifest['sites']
        self.assertEqual(first, second)
        self.assertEqual([site['ordinal'] for site in first], [1, 2])
        self.assertNotEqual(first[0]['id'], first[1]['id'])

    def test_device_callee_uses_same_context_and_has_no_entry_completion(self):
        result = instrument(HEADER + '.func f() { membar.sys; ret; } .entry k() { call f; ret; }')
        self.assertEqual(result.sources['module'].count('state+8], 1;'), 1)
        self.assertEqual(result.manifest['sites'][0]['function'], 'f')
        self.assertEqual(result.sources['module'].count('%globaltimer'), 2)

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected CUDA toolchain required; no GPU needed')
    def test_hopper_assembly_and_fence_timer_order(self):
        toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
        env = os.environ.copy()
        env.pop('LD_PRELOAD', None)
        env['CUDA_VISIBLE_DEVICES'] = ''
        sources = [HEADER + '.visible .entry k() { membar.sys; ret; }', HEADER + '''
.visible .entry k(.param .u64 ptr) {
 .reg .b64 %rd; .reg .b32 %r; .reg .pred %p;
 ld.param.u64 %rd, [ptr]; mov.u32 %r, %tid.x; setp.eq.u32 %p, %r, 0;
 @%p ld.acquire.sys.global.u32 %r, [%rd];
 @!%p st.release.sys.global.u32 [%rd], %r;
 atom.acq_rel.sys.global.add.u32 %r, [%rd], 1;
 st.global.u32 [%rd], %r; ret;
}
''']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, source in enumerate(sources):
                ptx, cubin = root / f'{index}.ptx', root / f'{index}.cubin'
                ptx.write_text(instrument(source).sources['module'])
                result = subprocess.run([str(toolchain / 'bin/ptxas'), '-arch=sm_90', str(ptx),
                                         '-o', str(cubin)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                sass = subprocess.check_output([str(toolchain / 'bin/cuobjdump'), '--dump-sass', str(cubin)],
                                               env=env, text=True)
                if index == 0:
                    lines = sass.splitlines()
                    timers = [i for i, line in enumerate(lines) if 'SR_GLOBALTIMER' in line]
                    fences = [i for i, line in enumerate(lines) if 'MEMBAR.SC.SYS' in line]
                    self.assertEqual(len(timers), 2, sass)
                    self.assertEqual(len(fences), 1, sass)
                    self.assertLess(timers[0], fences[0])
                    self.assertLess(fences[0], timers[1])


class GPUFixtureCPUTests(unittest.TestCase):
    def test_expected_all_thread_site_sequences(self):
        result = instrument_instructions(link_group(fixture_sources()), max_parameter_bytes=32764,
                                         interval_mode='instruction-only')
        sites = result.manifest['sites']
        self.assertEqual(len(sites), 7)
        self.assertEqual(len(expected_sites(sites, 0, False)), 8)
        self.assertEqual(len(expected_sites(sites, 1, False)), 7)
        self.assertEqual(expected_sites(sites, 0, False)[1:4], expected_sites(sites, 1, True)[:3])
        self.assertEqual([s['opcode'] for s in sites if s['operation'] == 'mbarrier'],
                         ['mbarrier.try_wait.shared.b64'])
        for source in fixture_sources().values():
            for function in parse(source).functions:
                for inst in function.instructions:
                    if inst.opcode.startswith('mbarrier.'):
                        self.assertIn(source[inst.start:inst.end], '\n'.join(result.sources.values()))
        self.assertFalse(any(s['memory_order'] in {'release', 'relaxed'} for s in sites))
        self.assertEqual({s['operation'] for s in sites},
                         {'ld', 'atom', 'fence', 'membar', 'mbarrier'})
        self.assertEqual(BufferLayout.from_dimensions(GRID, BLOCK).threads, 256)

    def test_gpu_cli_defaults_to_no_execution(self):
        result = subprocess.run([sys.executable, __file__, '--gpu'], capture_output=True, text=True,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['gpu_execution'])

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected CUDA toolchain required; no GPU needed')
    def test_multientry_fixture_assembles_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = compile_fixture(Path(temporary))
            self.assertEqual(set(report['binaries']), {'baseline', 'probed'})
            self.assertEqual(len(report['commands']), 9)
            self.assertFalse(report['probe']['runtime_verified'])

    @unittest.skipUnless(os.environ.get('UNIFIED_TOOLCHAIN'), 'Selected CUDA header/compiler required')
    def test_extended_launch_ctypes_layout_matches_actual_header(self):
        definitions = [(LaunchConfig, 'CUlaunchConfig'), (LaunchAttribute, 'CUlaunchAttribute'),
                       (LaunchAttributeValue, 'CUlaunchAttributeValue')]
        assertions = ['#include <cuda.h>', '#include <stddef.h>']
        for python, native in definitions:
            assertions.append(f'_Static_assert(sizeof({native}) == {C.sizeof(python)}, "size");')
            assertions.append(f'_Static_assert(_Alignof({native}) == {C.alignment(python)}, "alignment");')
            for name, _ in python._fields_:
                if name != 'alignment':
                    assertions.append(f'_Static_assert(offsetof({native}, {name}) == '
                                      f'{getattr(python, name).offset}, "offset");')
        for name, value in [('CU_LAUNCH_ATTRIBUTE_COOPERATIVE', 2),
                            ('CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH', 95),
                            ('CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT', 16),
                            ('CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT', 1),
                            ('CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT', 2)]:
            assertions.append(f'_Static_assert({name} == {value}, "enum");')
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'layout.c'
            source.write_text('\n'.join(assertions))
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
            env.pop('LD_PRELOAD', None)
            command = [os.environ['CC'], '-std=c11', '-fsyntax-only',
                       '-I' + str(Path(os.environ['UNIFIED_TOOLCHAIN']) / 'targets/x86_64-linux/include'), str(source)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


# GPU execution is opt-in and isolated in a guarded subprocess. Normal unittest
# discovery only parses/assembles these fixtures and never initializes CUDA.
GRID, BLOCK = (2, 2, 2), (8, 2, 2)
CALLEES = HEADER + '''
.visible .func direct(.param .u64 ptr) {
 .reg .b64 %addr, %barstate; .reg .b32 %value, %local, %dim, %baraddr;
 .reg .pred %done;
 .shared .align 8 .b64 barriers[1024];
 ld.param.u64 %addr, [ptr];
 membar.sys;
 membar.cta;
 fence.sc.sys;
 fence.sc.gpu;
 bar.sync 0;
 barrier.sync 0;
 bar.warp.sync 0xffffffff;
 mov.u32 %local, %tid.z; mov.u32 %dim, %ntid.y;
 mul.lo.u32 %local, %local, %dim; mov.u32 %dim, %tid.y;
 add.u32 %local, %local, %dim; mov.u32 %dim, %ntid.x;
 mul.lo.u32 %local, %local, %dim; mov.u32 %dim, %tid.x;
 add.u32 %local, %local, %dim;
 mov.u32 %baraddr, barriers; mad.lo.u32 %baraddr, %local, 8, %baraddr;
 mbarrier.init.shared.b64 [%baraddr], 1;
 mbarrier.arrive.shared.b64 %barstate, [%baraddr];
 mbarrier.test_wait.shared.b64 %done, [%baraddr], %barstate;
 @!%done trap;
 mbarrier.try_wait.shared.b64 %done, [%baraddr], %barstate;
 @!%done trap;
 mbarrier.inval.shared.b64 [%baraddr];
 ld.relaxed.sys.global.u32 %value, [%addr];
 add.u32 %value, %value, 1;
 st.relaxed.sys.global.u32 [%addr], %value; ret;
}
.visible .func first(.param .u64 ptr) {
 .reg .b64 %addr; .reg .b32 %value; ld.param.u64 %addr, [ptr];
 ld.acquire.sys.global.u32 %value, [%addr];
 add.u32 %value, %value, 3;
 st.release.sys.global.u32 [%addr], %value; ret;
}
.visible .func second(.param .u64 ptr) {
 .reg .b64 %addr; .reg .b32 %value; ld.param.u64 %addr, [ptr];
 atom.acq_rel.sys.global.add.u32 %value, [%addr], 7; ret;
}
'''
TABLE = HEADER + '''
.extern .func first(.param .u64 ptr);
.extern .func second(.param .u64 ptr);
.visible .global .align 8 .u64 callbacks[2] = {first, second};
.visible .global .align 4 .u32 invocations = 0;
'''


def fixture_sources(interval_mode='instruction-only'):
    # Independently flatten all six geometry dimensions in the application.
    flat = ''
    for register, coordinate, dimension in [('thread', 'tid', 'ntid'), ('block', 'ctaid', 'nctaid')]:
        flat += f'mov.u32 %{register}, %{coordinate}.z;\n'
        for axis in ('y', 'x'):
            flat += (f'mov.u32 %temp, %{dimension}.{axis}; mul.lo.u32 %{register}, %{register}, %temp;\n'
                     f'mov.u32 %temp, %{coordinate}.{axis}; add.u32 %{register}, %{register}, %temp;\n')
    caller = HEADER + '''
.extern .func direct(.param .u64 ptr);
.extern .global .align 8 .b8 callbacks[];
.extern .global .align 4 .u32 invocations;
.visible .entry dispatch(.param .align 16 .b8 blob[4096]) {
 .reg .b64 %base, %ptr, %offset, %table, %slot, %callee, %counter;
 .reg .b32 %thread, %block, %width, %temp, %bias, %value, %parity, %iteration;
 .reg .pred %even, %leader, %again;
 .param .u64 arg;
 .param .u64 indirect_arg;
 ld.param.u64 %base, [blob]; ld.param.u32 %bias, [blob+4092];
''' + flat + '''
 mov.u32 %width, %ntid.x;
 mov.u32 %temp, %ntid.y; mul.lo.u32 %width, %width, %temp;
 mov.u32 %temp, %ntid.z; mul.lo.u32 %width, %width, %temp;
 mad.lo.u32 %thread, %block, %width, %thread;
 mul.wide.u32 %offset, %thread, 4; add.u64 %ptr, %base, %offset;
 add.u32 %value, %thread, %bias; st.global.u32 [%ptr], %value;
 and.b32 %parity, %thread, 1; setp.eq.u32 %even, %parity, 0;
 @%even ld.acquire.sys.global.u32 %value, [%ptr];
 @!%even st.release.sys.global.u32 [%ptr], %value;
 st.param.u64 [arg], %ptr; call.uni direct, (arg);
 mul.wide.u32 %offset, %parity, 8;
 mov.u64 %table, callbacks; add.u64 %slot, %table, %offset;
 ld.global.u64 %callee, [%slot];
 st.param.u64 [indirect_arg], %ptr;
 { proto: .callprototype () _ (.param .u64 ptr); call %callee, (indirect_arg), proto; }
 mov.u32 %iteration, 0;
 LOOP: membar.sys;
 add.u32 %iteration, %iteration, 1; setp.lt.u32 %again, %iteration, 3; @%again bra LOOP;
 mov.u64 %counter, invocations; setp.eq.u32 %leader, %thread, 0;
 @%leader atom.global.add.u32 %temp, [%counter], 1;
 @%even ret; ret;
}
.visible .entry no_sites() { ret; }
'''
    if interval_mode == 'loop-boundary':
        caller = caller.replace('LOOP: membar.sys;', '''LOOP: membar.sys;
 .reg .u32 %inner; .reg .pred %inner_again;
 mov.u32 %inner, 0;
 INNER: fence.acq_rel.sys;
 add.u32 %inner, %inner, 1; setp.lt.u32 %inner_again, %inner, 17; @%inner_again bra INNER;
''')
    return {'caller': caller, 'table': TABLE, 'callees': CALLEES}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hashes():
    package = Path(__file__).resolve().parents[1] / 'neutrino/sys_sync'
    paths = [Path(__file__).resolve(), Path(__file__).with_name('test_sys_native.py').resolve(),
             package.parent / 'src/sys_launch.hpp', *sorted(package.glob('*.py'))]
    return {str(path): digest(path) for path in paths}


def compile_fixture(root, interval_mode='instruction-only', loop_depth=1, record_storage='per-thread-v1'):
    toolchain = Path(os.environ['UNIFIED_TOOLCHAIN'])
    env = os.environ.copy()
    env.pop('LD_PRELOAD', None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    sources = fixture_sources(interval_mode)
    probe = instrument_instructions(link_group(sources), max_parameter_bytes=32764,
                                    interval_mode=interval_mode, loop_depth=loop_depth, record_storage=record_storage)
    report = {'probe': probe.manifest, 'binaries': {}, 'commands': [], 'inputs': {
        'driver': str(Path(os.environ['NEUTRINO_REAL_DRIVER']).resolve()),
        'ptxas': str(toolchain / 'bin/ptxas'), 'nvlink': str(toolchain / 'bin/nvlink'),
        'cuda_header': str(toolchain / 'targets/x86_64-linux/include/cuda.h')}}
    report['input_hashes'] = {name: digest(path) for name, path in report['inputs'].items()}
    for variant, units in [('baseline', sources), ('probed', probe.sources)]:
        destination = root / variant
        destination.mkdir()
        objects = []
        for unit, source in units.items():
            ptx, obj = destination / f'{unit}.ptx', destination / f'{unit}.o'
            ptx.write_text(source)
            command = [str(toolchain / 'bin/ptxas'), '-arch=sm_90', '-c', str(ptx), '-o', str(obj)]
            with obj.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
            report['commands'].append(command)
            objects.append(str(obj))
        cubin = destination / 'linked.cubin'
        command = [str(toolchain / 'bin/nvlink'), '-arch=sm_90', *objects, '-o', str(cubin)]
        with cubin.with_suffix('.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        report['commands'].append(command)
        report['binaries'][variant] = {'path': str(cubin.relative_to(root)), 'sha256': digest(cubin)}
        sass = subprocess.check_output([str(toolchain / 'bin/cuobjdump'), '--dump-sass', str(cubin)],
                                       env=env, text=True, timeout=60)
        cubin.with_suffix('.sass').write_text(sass)
    bridge, command = compile_bridge(root, env)
    report['commands'].append(command)
    report['native_bridge'] = {'path': str(bridge.relative_to(root)), 'sha256': digest(bridge)}
    return report


class LaunchAttributeValue(C.Union):
    _fields_ = [('pad', C.c_char * 64), ('alignment', C.c_uint64), ('cooperative', C.c_int)]


class LaunchAttribute(C.Structure):
    _fields_ = [('id', C.c_int), ('pad', C.c_char * 4), ('value', LaunchAttributeValue)]


class LaunchConfig(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ('gridDimX', 'gridDimY', 'gridDimZ',
                 'blockDimX', 'blockDimY', 'blockDimZ', 'sharedMemBytes')] + [
        ('hStream', C.c_void_p), ('attrs', C.POINTER(LaunchAttribute)), ('numAttrs', C.c_uint)]


class FixtureDriver:
    """Small explicit Driver API fixture, not a production launch interceptor."""
    def __init__(self, path, expected_uuid, bridge):
        self.lib = C.CDLL(path)
        self.bridge = C.CDLL(str(bridge))
        self.bridge.fixturePrepare.argtypes = [C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.c_uint64]
        self.bridge.fixturePrepare.restype = C.c_void_p
        for name in ('fixtureParameters', 'fixtureExtra'):
            function = getattr(self.bridge, name)
            function.argtypes, function.restype = [C.c_void_p], C.POINTER(C.c_void_p)
        self.bridge.fixtureRelease.argtypes, self.bridge.fixtureRelease.restype = [C.c_void_p], None
        self.allocations, self.streams, self.modules = [], [], []
        self.context, self.device = C.c_void_p(), C.c_int()
        v, i, u, q, size = C.c_void_p, C.c_int, C.c_uint, C.c_uint64, C.c_size_t
        # Exact Linux 64-bit declarations from the selected CUDA header; these
        # versioned symbols deliberately bypass any future interposition layer.
        signatures = {
            'cuInit': [u], 'cuDriverGetVersion': [C.POINTER(i)],
            'cuDeviceGetCount': [C.POINTER(i)], 'cuDeviceGet': [C.POINTER(i), i],
            'cuDeviceGetUuid_v2': [v, i], 'cuDeviceComputeCapability': [C.POINTER(i), C.POINTER(i), i],
            'cuDeviceGetAttribute': [C.POINTER(i), i, i],
            'cuOccupancyMaxActiveBlocksPerMultiprocessor': [C.POINTER(i), v, i, size],
            'cuDevicePrimaryCtxRetain': [C.POINTER(v), i], 'cuDevicePrimaryCtxRelease_v2': [i],
            'cuCtxSetCurrent': [v], 'cuCtxSynchronize': [],
            'cuMemAlloc_v2': [C.POINTER(q), size], 'cuMemFree_v2': [q],
            'cuMemsetD8_v2': [q, C.c_ubyte, size], 'cuMemcpyHtoD_v2': [q, v, size],
            'cuMemcpyDtoH_v2': [v, q, size], 'cuModuleLoad': [C.POINTER(v), C.c_char_p],
            'cuModuleLoadData': [C.POINTER(v), v], 'cuModuleUnload': [v],
            'cuModuleGetFunction': [C.POINTER(v), v, C.c_char_p],
            'cuModuleGetGlobal_v2': [C.POINTER(q), C.POINTER(size), v, C.c_char_p],
            'cuStreamCreate': [C.POINTER(v), u], 'cuStreamDestroy_v2': [v],
            'cuLaunchKernel': [v, u, u, u, u, u, u, u, v, C.POINTER(v), C.POINTER(v)],
            'cuLaunchKernel_ptsz': [v, u, u, u, u, u, u, u, v, C.POINTER(v), C.POINTER(v)],
            'cuLaunchCooperativeKernel': [v, u, u, u, u, u, u, u, v, C.POINTER(v)],
            'cuLaunchKernelEx': [C.POINTER(LaunchConfig), v, C.POINTER(v), C.POINTER(v)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, i
        self.call('cuInit', 0)
        count, major, minor, version = i(), i(), i(), i()
        self.call('cuDeviceGetCount', C.byref(count))
        if count.value != 1:
            raise RuntimeError('The guarded fixture requires exactly one visible GPU')
        self.call('cuDeviceGet', C.byref(self.device), 0)
        physical = C.create_string_buffer(16)
        self.call('cuDeviceGetUuid_v2', physical, self.device)
        self.uuid = 'GPU-' + str(uuid.UUID(bytes=physical.raw))
        self.call('cuDeviceComputeCapability', C.byref(major), C.byref(minor), self.device)
        self.call('cuDriverGetVersion', C.byref(version))
        self.version = version.value
        if self.uuid != expected_uuid or (major.value, minor.value) != (9, 0):
            raise RuntimeError('GPU UUID/architecture changed before context creation')
        self.call('cuDevicePrimaryCtxRetain', C.byref(self.context), self.device)
        try:
            self.call('cuCtxSetCurrent', self.context)
        except Exception:
            self.close()
            raise

    def call(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            raise RuntimeError(f'{name} failed with CUDA error {result}; no launch retry')

    def allocate(self, size):
        pointer = C.c_uint64()
        self.call('cuMemAlloc_v2', C.byref(pointer), size)
        self.allocations.append(pointer.value)
        self.call('cuMemsetD8_v2', pointer, 0, size)
        return pointer.value

    def write(self, pointer, data):
        data = C.create_string_buffer(data, len(data))
        self.call('cuMemcpyHtoD_v2', pointer, data, C.sizeof(data))

    def read(self, pointer, size):
        data = C.create_string_buffer(size)
        self.call('cuMemcpyDtoH_v2', data, pointer, size)
        return data.raw

    def load(self, path, memory=False):
        module = C.c_void_p()
        if memory:
            data = C.create_string_buffer(path.read_bytes())
            self.call('cuModuleLoadData', C.byref(module), data)
        else:
            self.call('cuModuleLoad', C.byref(module), os.fsencode(path))
        self.modules.append(module)
        return module

    def symbol(self, module, name, size):
        pointer, actual = C.c_uint64(), C.c_size_t()
        self.call('cuModuleGetGlobal_v2', C.byref(pointer), C.byref(actual), module, name.encode())
        if actual.value != size:
            raise RuntimeError(f'Unexpected global size: {name}')
        return pointer.value

    def stream(self):
        stream = C.c_void_p()
        self.call('cuStreamCreate', C.byref(stream), 1)
        self.streams.append(stream)
        return stream.value

    @contextmanager
    def marshalled(self, entry, arguments, api, instrumented):
        original = arguments[:-1] if instrumented else arguments
        parameters = (C.c_void_p * len(original))(*(C.addressof(arg) for arg in original))
        extra = None
        if api == 'packed':
            packed = b''.join(C.string_at(C.byref(arg), C.sizeof(arg)) for arg in original)
            data, size = C.create_string_buffer(packed, len(packed)), C.c_size_t(len(packed))
            extra = (C.c_void_p * 5)(1, C.addressof(data), 2, C.addressof(size), 0)
            parameters = None
        if not instrumented:
            yield parameters, extra
            return
        handle = self.bridge.fixturePrepare(int(entry == 'dispatch'), parameters, extra, arguments[-1].value)
        if not handle:
            raise RuntimeError('Native launch argument preparation failed; no launch submitted')
        try:
            yield self.bridge.fixtureParameters(handle), self.bridge.fixtureExtra(handle)
        finally:
            self.bridge.fixtureRelease(handle)

    def launch(self, module, entry, stream, arguments, api, instrumented):
        function = C.c_void_p()
        self.call('cuModuleGetFunction', C.byref(function), module, entry.encode())
        resource = {}
        if api in ('cooperative', 'extended-cooperative'):
            active, multiprocessors, supported = C.c_int(), C.c_int(), C.c_int()
            self.call('cuDeviceGetAttribute', C.byref(supported), 95, self.device)
            self.call('cuDeviceGetAttribute', C.byref(multiprocessors), 16, self.device)
            self.call('cuOccupancyMaxActiveBlocksPerMultiprocessor', C.byref(active), function, 32, 0)
            if not supported.value or active.value * multiprocessors.value < 8:
                raise RuntimeError('Instrumented fixture exceeds cooperative residency limits')
            resource = dict(active_blocks_per_sm=active.value, multiprocessors=multiprocessors.value,
                            cooperative_grid_blocks=8)
        with self.marshalled(entry, arguments, api, instrumented) as (parameters, extra):
            if api == 'cooperative':
                self.call('cuLaunchCooperativeKernel', function, *GRID, *BLOCK, 0, stream, parameters)
            elif api == 'extended-cooperative':
                attribute = LaunchAttribute()
                attribute.id, attribute.value.cooperative = 2, 1
                config = LaunchConfig(*GRID, *BLOCK, 0, stream, C.pointer(attribute), 1)
                self.call('cuLaunchKernelEx', C.byref(config), function, parameters, extra)
            elif api in ('ordinary', 'ptsz', 'packed'):
                symbol = 'cuLaunchKernel_ptsz' if api == 'ptsz' else 'cuLaunchKernel'
                self.call(symbol, function, *GRID, *BLOCK, 0, stream, parameters, extra)
            else:
                raise RuntimeError(f'Unsupported fixture launch API: {api}')
        return resource

    def close(self):
        errors = []
        operations = [('cuCtxSynchronize', ())]
        operations += [('cuStreamDestroy_v2', (s,)) for s in reversed(self.streams)]
        operations += [('cuModuleUnload', (m,)) for m in reversed(self.modules)]
        operations += [('cuMemFree_v2', (p,)) for p in reversed(self.allocations)]
        operations += [('cuCtxSetCurrent', (None,)), ('cuDevicePrimaryCtxRelease_v2', (self.device,))]
        for name, arguments in operations:
            try:
                self.call(name, *arguments)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('Fixture teardown failed: ' + '; '.join(errors))


def expected_sites(sites, thread, swapped, interval_mode='instruction-only', loop_depth=1):
    functions = {}
    for site in sites:
        functions.setdefault(site['function'], []).append(site['ordinal'])
    entry = functions['dispatch']
    callee = 'second' if bool(thread & 1) != swapped else 'first'
    tail = ([entry[1], *([entry[2]] * 3)] if loop_depth == 1 else [entry[1]]) \
        if interval_mode == 'loop-boundary' else [entry[1]] * 3
    head = [] if thread & 1 else [entry[0]]
    return [*head, *functions['direct'], *functions[callee], *tail]


def fixture_worker(root):
    report = {'schema': 1, 'status': 'running', 'kind': 'instruction_probe_gpu_fixture',
              'native_interception_validated': False, 'polling_intervals_validated': False,
              'cross_library_capture_validated': False, 'launches': []}
    try:
        setup = json.loads((root / 'launch.json').read_text())
        if (os.environ.get('UNIFIED_PROBE_WORKER') != setup['session'] or
                os.environ.get('CUDA_VISIBLE_DEVICES') != setup['device']['uuid'] or
                implementation_hashes() != setup['implementation_hashes']):
            raise RuntimeError('GPU fixture must be started by its guarded parent')
        build = setup['build']
        for name, path in build['inputs'].items():
            if digest(path) != build['input_hashes'][name]:
                raise RuntimeError('Fixture compiler/driver evidence changed')
        for binary in [*build['binaries'].values(), build['native_bridge']]:
            if digest(root / binary['path']) != binary['sha256']:
                raise RuntimeError('Fixture image changed before launch')
        driver = FixtureDriver(build['inputs']['driver'], setup['device']['uuid'], root / build['native_bridge']['path'])
        try:
            registry = Registry(ProcessIdentity(setup['session'], os.getpid(), uuid.uuid4().hex))
            registry.context_created(driver.context.value, driver.uuid)
            report.update(gpu_uuid=driver.uuid, driver_version=driver.version, session=setup['session'])
            original = ArtifactIdentity(build['binaries']['baseline']['sha256'], 'sm_90', 'linked_group')
            modules = []
            for variant, memory in [('baseline', False), ('probed', True), ('probed', False)]:
                module = driver.load(root / build['binaries'][variant]['path'], memory)
                registry.module_loaded(driver.context.value, module.value, original)
                modules.append(module)
            streams = [driver.stream(), driver.stream()]
            sites = build['probe']['sites']
            retained = 0

            def prepare(index, bias, *, capacity=16, enabled=True, swapped=False, entry='dispatch',
                        api='ordinary'):
                nonlocal retained
                layout = BufferLayout.from_dimensions(GRID, BLOCK, capacity)
                layout.require_budget(retained)
                retained += layout.device_bytes
                output = driver.allocate(layout.threads * 4)
                context = driver.allocate(layout.device_bytes)
                states = context + CONTEXT.size
                records = states + layout.state_bytes
                module, stream = modules[index], 0 if api == 'ptsz' else streams[bias % 2]
                kernel = registry.kernel(driver.context.value, module.value, entry)
                launch = registry.launch(kernel, stream)
                driver.write(context, CONTEXT.pack(records, states, launch.ordinal, layout.threads,
                                                    layout.capacity, 3, 0, 0))
                blob = bytearray(4096)
                struct.pack_into('<Q', blob, 0, output)
                struct.pack_into('<I', blob, 4092, bias)
                arguments = [C.create_string_buffer(bytes(blob), 4096)] if entry == 'dispatch' else []
                if index:
                    arguments.append(C.c_uint64(context if enabled else 0))
                return dict(index=index, bias=bias, enabled=enabled and index != 0, swapped=swapped,
                            entry=entry, api=api, module=module, stream=stream, arguments=arguments, launch=launch,
                            layout=layout, output=output, states=states, context=context)

            def execute(batch):
                # All allocations/initialization precede the asynchronous batch.
                # No readback or stream synchronization occurs between launches.
                for job in batch:
                    job['resources'] = driver.launch(job['module'], job['entry'], job['stream'],
                                                     job['arguments'], job['api'], job['index'] != 0)
                driver.call('cuCtxSynchronize')
                for job in batch:
                    layout, launch = job['layout'], job['launch']
                    values = list(struct.unpack(f'<{layout.threads}I', driver.read(job['output'], layout.threads * 4)))
                    expected = [t + job['bias'] + 1 + (7 if bool(t & 1) != job['swapped'] else 3)
                                for t in range(layout.threads)] if job['entry'] == 'dispatch' else [0] * layout.threads
                    (root / f'launch-{launch.ordinal}-values.json').write_text(json.dumps({
                        'index': job['index'], 'entry': job['entry'], 'actual': values, 'expected': expected}))
                    if values != expected:
                        mismatch = next(t for t in range(layout.threads) if values[t] != expected[t])
                        raise RuntimeError(f'Launch {launch.ordinal} module {job["index"]} thread {mismatch}: '
                                           f'actual={values[mismatch]}, expected={expected[mismatch]}')
                    raw = encode_header(launch, layout) + driver.read(job['states'], layout.state_bytes + layout.record_bytes)
                    (root / f'launch-{launch.ordinal}.bin').write_bytes(raw)
                    row = {key: job[key] for key in ('index', 'bias', 'enabled', 'swapped', 'entry', 'api', 'resources')}
                    row.update(launch_id=launch.id, identity=asdict(launch), grid=GRID, block=BLOCK,
                               layout=asdict(layout), raw=f'launch-{launch.ordinal}.bin', numerical_match=True)
                    if not job['enabled']:
                        if any(raw[RAW_HEADER.size:]):
                            raise RuntimeError('Null/baseline context unexpectedly recorded data')
                        row['records'] = 0
                    elif layout.capacity == 2:
                        for thread in range(layout.threads):
                            if STATE.unpack_from(raw, RAW_HEADER.size + thread * STATE.size) != (2, 1, 1, 0):
                                raise RuntimeError('Overflow wrapped, skipped completion, or was not reported')
                        try:
                            decode(raw, launch=launch, layout=layout, sites=[s['ordinal'] for s in sites])
                        except RecordError as exc:
                            if 'overflow' not in str(exc).lower():
                                raise
                        else:
                            raise RuntimeError('Overflow was accepted by the decoder')
                        row['expected_overflow_rejected'] = True
                    else:
                        intervals = decode(raw, launch=launch, layout=layout, sites=[s['ordinal'] for s in sites],
                                           loop_sites=[s['ordinal'] for s in sites if s.get('interval_kind') == 'loop'])
                        per_thread = [[] for _ in range(layout.threads)]
                        for interval in intervals:
                            per_thread[interval.thread].append(interval.site)
                        for thread, observed in enumerate(per_thread):
                            expected = expected_sites(sites, thread, job['swapped'], build['probe']['interval_mode'],
                                                      build['probe'].get('loop_depth', 1)) \
                                if job['entry'] == 'dispatch' else []
                            if observed != expected:
                                raise RuntimeError(f'Thread/site/predicate mismatch on thread {thread}')
                        row['records'] = len(intervals)
                        (root / f'launch-{launch.ordinal}.json').write_text(json.dumps([asdict(i) for i in intervals]))
                    report['launches'].append(row)

            execute([prepare(0, 10)])
            execute([prepare(1, 20), prepare(1, 21), prepare(2, 22)])
            table = driver.symbol(modules[1], 'callbacks', 16)
            before = driver.read(table, 16)
            driver.write(table, before[8:] + before[:8])
            execute([prepare(1, 30, swapped=True), prepare(2, 31)])
            execute([prepare(1, 40, enabled=False, swapped=True), prepare(1, 41, capacity=2, swapped=True),
                     prepare(1, 42, entry='no_sites')])
            for bias, api in enumerate(('packed', 'ptsz', 'cooperative', 'extended-cooperative'), 50):
                execute([prepare(0, bias, api=api), prepare(2, bias, api=api)])
            counts = [struct.unpack('<I', driver.read(driver.symbol(m, 'invocations', 4), 4))[0] for m in modules]
            if counts != [5, 5, 6]:
                raise RuntimeError(f'Original launch count/module-global ownership mismatch: {counts}')
            report.update(invocations=counts, launches_submitted=17, retained_trace_bytes=retained,
                          same_symbol_module_isolation=True, concurrent_stream_submission=True,
                          concurrent_execution_proven=False, all_thread_records_validated=True,
                          native_argument_marshalling_validated=True)
        finally:
            driver.close()
        report.update(status='passed', teardown_completed=True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    (root / 'worker.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'passed' else 1


def gpu_main():
    parser = argparse.ArgumentParser(description='Explicit guarded instruction-probe GPU fixture')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allocation-confirmed', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--reconstruction', type=Path)
    parser.add_argument('--interval-mode', choices=('instruction-only', 'loop-boundary'), default='instruction-only')
    parser.add_argument('--loop-depth', type=int, default=1)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.output or not os.environ.get('UNIFIED_PROBE_WORKER'):
            parser.error('Worker requires its guarded parent')
        return fixture_worker(args.output.resolve())
    if not args.execute:
        print(json.dumps({'gpu_execution': False, 'kind': 'instruction_probe_gpu_fixture',
                          'grid': GRID, 'block': BLOCK, 'native_interception_validated': False}))
        return 0
    if not args.allocation_confirmed or not all((args.output, args.preflight, args.reconstruction)):
        parser.error('Execution requires allocation confirmation, fresh output, preflight, and reconstruction')
    workspace = Path(__file__).resolve().parents[2]
    root = args.output.resolve()
    if not root.is_relative_to(workspace / 'artifacts/unified-sys') or root.exists():
        parser.error('Use a fresh output directory under artifacts/unified-sys')
    if Path(sys.prefix).resolve() != (workspace / '.venv-unified').resolve():
        parser.error('Source examples/moe_training/env.sh and use the isolated Python')
    sys.path.insert(0, str(workspace / 'neutrino/examples/moe_training'))
    from gpu_guard import check_once, launch_lock, require_idle
    from launch import require_clean_baseline, validate_evidence
    from setup_environment import verify_sources
    from config import Workload
    require_clean_baseline(os.environ)
    sources = verify_sources()
    identity = validate_evidence(json.loads(args.preflight.read_text()), json.loads(args.reconstruction.read_text()),
                                 args.reconstruction.resolve().parent, sources, Workload())
    root.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'status': 'running', 'kind': 'instruction_probe_gpu_fixture',
              'cross_library_capture_validated': False, 'native_interception_validated': False,
              'sources': sources, 'environment_identity': identity}
    try:
        build = compile_fixture(root, args.interval_mode, args.loop_depth)
        with launch_lock() as lock:
            devices = require_idle()
            session = uuid.uuid4().hex
            setup = dict(session=session, device=devices[0], build=build,
                         implementation_hashes=implementation_hashes())
            (root / 'launch.json').write_text(json.dumps(setup, indent=2) + '\n')
            final = check_once()
            if [(d['index'], d['uuid']) for d in final] != [(d['index'], d['uuid']) for d in devices]:
                raise RuntimeError('GPU identities changed before launching fixture')
            env = os.environ.copy()
            env.pop('LD_PRELOAD', None)
            env.update(CUDA_VISIBLE_DEVICES=devices[0]['uuid'], UNIFIED_PROBE_WORKER=session)
            command = [sys.executable, str(Path(__file__).resolve()), '--gpu', '--worker', '--output', str(root)]
            with (root / 'console.log').open('w') as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=120, pass_fds=(lock.fileno(),))
            report.update(exit_code=process.returncode, device=devices[0], command=command)
            worker = json.loads((root / 'worker.json').read_text())
            if process.returncode or worker['status'] != 'passed':
                raise RuntimeError(f'GPU fixture failed: {worker.get("error", process.returncode)}')
            report['worker'] = 'worker.json'
            report['source_integrity_after'] = verify_sources()
            if implementation_hashes() != setup['implementation_hashes']:
                raise RuntimeError('Fixture/tracer source changed during execution')
            validate_evidence(json.loads(args.preflight.read_text()), json.loads(args.reconstruction.read_text()),
                              args.reconstruction.resolve().parent, sources, Workload())
            report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        raise SystemExit(gpu_main())
    unittest.main()
