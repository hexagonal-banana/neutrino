"""CPU-only workload and adapter contracts; fake transport, real CPU autograd."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.moe_training.config import Workload
from examples.moe_training import adapter


class WorkloadTests(unittest.TestCase):
    def test_original_topology_and_capture(self):
        manifest = Workload().manifest()
        self.assertEqual((manifest['dense_dp'], manifest['expert_parallel_size'],
                          manifest['expert_dp'], manifest['tensor_parallel_size']), (8, 8, 1, 1))
        self.assertEqual(manifest['gradient_accumulation_steps'], 1)
        self.assertEqual((manifest['capture_first_step'], manifest['capture_last_step']), (3, 12))
        self.assertEqual(manifest['train_iters'], 12)
        self.assertEqual(manifest['capture_last_step'] - manifest['capture_first_step'] + 1, 10)
        argv = Workload().megatron_args()
        self.assertIn('--use-distributed-optimizer', argv)
        self.assertEqual(argv[argv.index('--rerun-mode') + 1], 'disabled')
        self.assertEqual(argv[argv.index('--moe-flex-dispatcher-backend') + 1], 'deepepv2')
        self.assertNotIn('--profile', argv)
        self.assertNotIn('--cuda-graph-impl', argv)

    def test_two_rank_fixture(self):
        self.assertEqual(replace(Workload(), world_size=2, expert_parallel_size=2,
                                 global_batch_size=2).manifest()['dense_dp'], 2)

    def test_mixed_topology_and_sequence_parallel(self):
        workload = replace(Workload(), tensor_parallel_size=2, expert_parallel_size=4, global_batch_size=4)
        manifest = workload.manifest()
        self.assertEqual((manifest['dense_dp'], manifest['tensor_parallel_size'],
                          manifest['expert_parallel_size'], manifest['expert_dp']), (4, 2, 4, 2))
        self.assertEqual(manifest['gradient_accumulation_steps'], 1)
        argv = workload.megatron_args()
        self.assertIn('--sequence-parallel', argv)
        self.assertEqual(argv[argv.index('--tensor-model-parallel-size') + 1], '2')
        self.assertEqual(argv[argv.index('--expert-model-parallel-size') + 1], '4')

    def test_invalid_dimensions(self):
        cases = [{'tensor_parallel_size': 3}, {'expert_tensor_parallel_size': 2},
                 {'expert_parallel_size': 3}, {'world_size': True}, {'micro_batch_size': 0},
                 {'global_batch_size': 7}, {'hidden_size': 1537}, {'num_experts': 7},
                 {'topk': 9}, {'capture_last_step': 13}, {'capture_first_step': 13},
                 {'learning_rate': float('nan')}]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(Workload(), **values).validate()

    def test_config_import_does_not_load_frameworks(self):
        command = [sys.executable, '-c',
                   'from examples.moe_training.config import Workload; import sys; '
                   'Workload().megatron_args(); '
                   'assert not ({"torch", "deep_ep", "megatron"} & set(sys.modules))']
        subprocess.run(command, cwd=ROOT, check=True)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        torch = self.torch
        self.cuda_guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU forbidden'))
        self.cuda_guard.start()
        self.addCleanup(self.cuda_guard.stop)
        self.calls = []
        calls = self.calls

        class Buffer:
            def __init__(self, group=None, **kwargs):
                self.num_max_tokens_per_rank = kwargs.get('num_max_tokens_per_rank', 4)
                self.destroyed = False

            def capture(self):
                calls.append(('capture', {}))
                return 'previous-event'

            def dispatch(self, x, *, do_expand, do_cpu_sync, defer_epilogue, handle=None, **kwargs):
                calls.append(('dispatch', dict(kwargs, handle=handle, do_expand=do_expand,
                                                do_cpu_sync=do_cpu_sync, defer_epilogue=defer_epilogue)))
                if handle is None:
                    handle = SimpleNamespace(num_sms=4, num_recv_tokens_per_expert_list=[x.shape[0]])
                done = SimpleNamespace(current_stream_wait=lambda: calls.append(('wait', {})))
                return (x.clone(), kwargs.get('topk_idx'), kwargs.get('topk_weights'), handle, done)

            def combine(self, x, **kwargs):
                calls.append(('combine', dict(kwargs, x=x.clone())))
                done = SimpleNamespace(current_stream_wait=lambda: calls.append(('wait', {})))
                return x.clone(), kwargs.get('topk_weights'), done

            def destroy(self):
                self.destroyed = True

        self.Buffer = Buffer
        fake_ep = ModuleType('deep_ep')
        fake_ep.__version__ = '2.5.0'
        fake_ep.topk_idx_t = torch.int64
        fake_ep.EPBuffer = Buffer
        modules = {name: ModuleType(name) for name in
                   ('megatron', 'megatron.core', 'megatron.core.transformer',
                    'megatron.core.transformer.moe', 'megatron.core.transformer.moe.fused_a2a',
                    'megatron.core.transformer.moe.token_dispatcher')}
        self.fused = modules['megatron.core.transformer.moe.fused_a2a']
        self.fused.__file__ = str(ROOT / 'fake_megatron/fused_a2a.py')
        self.modules = patch.dict(sys.modules, dict(modules, deep_ep=fake_ep))
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.git = patch.object(adapter.subprocess, 'check_output', side_effect=lambda argv, **kw:
                                adapter.MEGATRON_REVISION if 'rev-parse' in argv else '')
        self.git_mock = self.git.start()
        self.addCleanup(self.git.stop)
        self.api = adapter.install(ROOT / 'fake_megatron')

    def dispatch(self, *, async_finish=False):
        torch = self.torch
        x = torch.randn(4, 8, requires_grad=True)
        weights = torch.randn(4, 2, requires_grad=True)
        indices = torch.zeros(4, 2, dtype=torch.int32)
        outputs = self.api['dispatch'](self.Buffer(), x, indices, weights, 8, 4,
                                       async_finish=async_finish,
                                       allocate_on_comm_stream=async_finish)
        return x, weights, outputs

    def test_input_copy_precedes_comm_stream_event(self):
        torch = self.torch
        x = torch.randn(8, 4).t()
        self.assertFalse(x.is_contiguous())
        original = torch.Tensor.contiguous

        def contiguous(tensor, *args, **kwargs):
            if tensor is x:
                self.calls.append(('input_copy', {}))
            return original(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, 'contiguous', contiguous):
            self.api['dispatch'](self.Buffer(), x, torch.zeros(4, 2, dtype=torch.int64),
                                 torch.ones(4, 2), 8, 4, async_finish=True, allocate_on_comm_stream=True)
        order = [name for name, _ in self.calls]
        self.assertLess(order.index('input_copy'), order.index('capture'))

    def test_optional_weight_gradient(self):
        x, weights, (rx, _, _, _, _) = self.dispatch()
        rx.sum().backward()
        self.assertTrue(self.torch.equal(x.grad, self.torch.ones_like(x)))
        self.assertIsNone(weights.grad)
        self.assertIsNone(self.calls[-1][1]['topk_weights'])

    def test_weight_only_gradient_materializes_zero_tokens(self):
        x, weights, (_, _, rw, _, _) = self.dispatch()
        rw.sum().backward()
        self.assertTrue(self.torch.equal(weights.grad, self.torch.ones_like(weights)))
        self.assertTrue(self.torch.equal(x.grad, self.torch.zeros_like(x)))

    def test_roundtrip_backward_uses_cached_layout(self):
        x, weights, (rx, _, rw, _, handle) = self.dispatch(async_finish=True)
        result, result_weights = self.api['combine'](self.Buffer(), rx, handle,
                                                    async_finish=True, allocate_on_comm_stream=True)
        self.assertIsNone(result_weights)
        (result.sum() + rw.sum()).backward()
        self.assertTrue(all(value == 1 for value in self.api['counts'].values()))
        dispatches = [values for name, values in self.calls if name == 'dispatch']
        self.assertIs(dispatches[1]['handle'], handle)
        self.assertFalse(dispatches[1]['do_cpu_sync'])
        self.assertFalse(dispatches[1]['do_expand'])
        self.assertFalse(dispatches[1]['defer_epilogue'])
        self.assertEqual(dispatches[0]['topk_idx'].dtype, self.torch.int64)
        self.assertEqual(dispatches[0]['topk_weights'].dtype, self.torch.float32)
        self.assertEqual(len([name for name, _ in self.calls if name == 'wait']), 4)

    def test_version_pin_and_idempotence(self):
        self.assertIs(adapter.install(ROOT / 'fake_megatron'), self.api)
        self.git_mock.side_effect = lambda *args, **kwargs: 'incorrect-revision'
        with self.assertRaisesRegex(RuntimeError, 'pinned Megatron'):
            adapter.install(ROOT / 'fake_megatron')


def annotate_parallel_roles(document, coverage, reports):
    """Attach conservative application-side role evidence; never infer TP from opcodes."""
    from neutrino.sys_sync.export import normalize_chrome_document
    launches = {row['launch_id']: row for row in coverage if row.get('launch_id')}
    origin = document['metadata']['origin_host_ns']
    events = []
    for original in document['traceEvents']:
        if original.get('ph') == 'M' and original.get('name') == 'thread_name':
            continue
        event = {**original, 'args': dict(original.get('args', {}))}
        if event.get('cat', '').endswith('.kernel'):
            args = event['args']
            row = launches[args['launch_id']]
            role, evidence = 'EP', {'method': 'DeepEP image identity'}
            if args['library'] == 'NCCL':
                matches = [call for call in reports[row['rank']]['collective_calls']
                           if call['step'] == row['step'] and call['successful'] and
                           call['begin_host_ns'] <= row['host_ns'] <= call['end_host_ns']]
                if len(matches) == 1:
                    call = matches[0]
                    role, evidence = call['role'], {'method': 'unique host API submission bracket', **call}
                else:
                    role, evidence = 'Unattributed', {'method': 'no unique host API submission bracket',
                                                     'candidate_count': len(matches)}
            event['cat'] = f'{role}.{args["library"]}.kernel'
            event['tid'] = f'{role} {args["library"]}/stream{args["cupti_stream"]}'
            event['name'] = role + ' ' + event['name']
            args['parallel_role'] = role
            args['role_evidence'] = evidence
        elif event.get('ph') == 'X':
            event['tid'] = 'host capture'
        events.append(event)
    for rank, report in reports.items():
        for call in report['collective_calls']:
            events.append({'ph': 'X', 'name': call['role'] + ' ' + call['api'],
                           'cat': call['role'] + '.host_collective', 'pid': report['pid'],
                           'tid': f'host collectives/thread{call["host_thread"]}',
                           'ts': (call['begin_host_ns'] - origin) / 1000,
                           'dur': (call['end_host_ns'] - call['begin_host_ns']) / 1000,
                           'args': {'rank': rank, **call}})
    result = normalize_chrome_document({**document, 'traceEvents': events})
    result['metadata'] = {**document['metadata'],
                          'role_attribution': 'Unique host API submission brackets with actual process-group metadata; unmatched kernels remain unattributed'}
    return result


class ParallelRoleTests(unittest.TestCase):
    def test_collective_audit_preserves_calls_and_errors(self):
        from contextlib import ExitStack, contextmanager
        import time
        import torch.distributed as dist
        from examples.moe_training.worker import install_collective_audit
        group = SimpleNamespace(group_name='tp', group_desc='TENSOR_MODEL_PARALLEL_GROUP')
        calls, report = [], {}

        def all_reduce(tensor, op=None, group=None, async_op=False):
            calls.append((tensor, group, async_op))
            if tensor == 'fail':
                raise ValueError('original error')
            return 'original result'

        exit_ticks, suppress = [], False

        @contextmanager
        def coalescing(group=None, device=None, async_ops=False):
            try:
                yield 'coalesced result'
            except ValueError:
                if not suppress:
                    raise
            finally:
                exit_ticks.append(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))

        names = ('all_reduce', 'all_gather', 'all_gather_into_tensor', '_all_gather_base',
                 'reduce_scatter', 'reduce_scatter_tensor', '_reduce_scatter_base',
                 'broadcast', 'all_to_all_single')
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(dist, name, all_reduce))
            stack.enter_context(patch.object(dist, '_coalescing_manager', coalescing))
            stack.enter_context(patch.object(dist, 'get_backend', return_value='nccl'))
            stack.enter_context(patch.object(dist, 'get_process_group_ranks', return_value=[0, 1]))
            state, wrap = install_collective_audit(report)
            self.assertIs(wrap(dist.all_reduce), dist.all_reduce)
            self.assertEqual(dist.all_reduce('inactive', group=group), 'original result')
            self.assertEqual(report['collective_calls'], [])
            state['step'] = 3
            self.assertEqual(dist.all_reduce('active', group=group, async_op=True), 'original result')
            with self.assertRaisesRegex(ValueError, 'original error'):
                dist.all_reduce('fail', group=group)
            self.assertEqual(len(calls), 3)
            successful, failed = report['collective_calls']
            self.assertTrue(successful['successful'])
            self.assertTrue(successful['async_op'])
            self.assertFalse(failed['successful'])
            self.assertEqual(successful['role'], 'TP')
            self.assertEqual(successful['group_ranks'], [0, 1])
            self.assertLessEqual(successful['begin_host_ns'], successful['end_host_ns'])
            with dist._coalescing_manager(group, async_ops=True) as result:
                self.assertEqual(result, 'coalesced result')
                body_tick = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            row = report['collective_calls'][-1]
            self.assertEqual(row['submission_phase'], 'context_exit')
            self.assertTrue(row['async_op'])
            self.assertLessEqual(body_tick, row['begin_host_ns'])
            self.assertLessEqual(row['begin_host_ns'], exit_ticks[-1])
            self.assertLessEqual(exit_ticks[-1], row['end_host_ns'])
            with self.assertRaisesRegex(ValueError, 'body error'):
                with dist._coalescing_manager(group):
                    raise ValueError('body error')
            suppress = True
            with dist._coalescing_manager(group):
                raise ValueError('suppressed by original manager')

    def test_group_description_not_collective_name(self):
        from examples.moe_training.worker import collective_role
        self.assertEqual(collective_role('TENSOR_MODEL_PARALLEL_GROUP'), 'TP')
        self.assertEqual(collective_role('EXPERT_DATA_PARALLEL_GROUP'), 'DP')
        self.assertEqual(collective_role('DATA_PARALLEL_GROUP_WITH_CP'), 'DP')
        self.assertEqual(collective_role('MODEL_PARALLEL_GROUP'), 'Other')

    def test_unique_missing_and_ambiguous_submission_brackets(self):
        import copy
        call = dict(step=3, successful=True, begin_host_ns=10, end_host_ns=20, role='TP',
                    api='all_reduce', host_thread=7, group_ranks=[0, 1])
        document = {'metadata': {'origin_host_ns': 0}, 'traceEvents': [
            {'ph': 'X', 'cat': 'DP.NCCL.kernel', 'pid': 10, 'tid': 1, 'ts': 1, 'dur': 2,
             'name': 'NCCL AllReduce', 'args': {'launch_id': 'L', 'library': 'NCCL', 'cupti_stream': 2}}]}
        original = copy.deepcopy(document)
        for calls, expected in (([call], 'TP'), ([], 'Unattributed'), ([call, call], 'Unattributed')):
            report = {0: {'pid': 10, 'collective_calls': calls}}
            result = annotate_parallel_roles(document, [dict(launch_id='L', rank=0, step=3, host_ns=15)], report)
            kernel = next(e for e in result['traceEvents'] if e.get('cat', '').endswith('.kernel'))
            self.assertEqual(kernel['cat'], expected + '.NCCL.kernel')
            self.assertEqual((kernel['ts'], kernel['dur']), (1, 2))
            self.assertTrue(all(type(e['tid']) is int for e in result['traceEvents']))
        self.assertEqual(document, original)


def validate_training_capture(directory, decoded_subdir='decoded'):
    """Audit the actual ten-step deliverable, not just a successful worker exit."""
    import collections
    import hashlib
    import json
    import math
    from neutrino.sys_sync.export import load_processes, normalize_chrome_document

    directory = Path(directory).resolve()
    read = lambda name: json.loads((directory / name).read_text())
    decoded_directory = (directory / decoded_subdir).resolve()
    if not decoded_directory.is_relative_to(directory):
        raise ValueError('Decoded output must belong to the capture directory')
    decoded_read = lambda name: json.loads((decoded_directory / name).read_text())
    launch, execution, decoded = read('launch.json'), read('validation.json'), decoded_read('validation.json')
    workload = Workload(**launch['workload']).validate()
    assert workload.world_size == 8
    mixed = workload.tensor_parallel_size > 1
    assert (workload.expert_parallel_size, workload.tensor_parallel_size) in ((8, 1), (4, 2))
    steps = list(range(workload.capture_first_step, workload.capture_last_step + 1))
    assert len(steps) == 10 and steps == list(range(3, 13))
    assert execution['status'] == 'passed' and execution['trace'] and execution['mode'] == 'train'
    assert execution['rank_status'] == ['passed'] * 8
    assert decoded['full_trace_acceptance'] and decoded['kernel_activity_verified']
    assert decoded['steps'] == steps and decoded['ranks'] == list(range(8))
    processes = load_processes(directory / 'raw', range(8), steps)
    assert all(p.activity_verified and p.identity.session == launch['session'] for p in processes)
    policy = launch['native_runtime'].get('capture_policy')
    if policy is not None:
        from neutrino.sys_sync.sampling import policy_hash
        assert all(p.capture_policy == policy and p.process['capture_policy_sha256'] == policy_hash(policy)
                   for p in processes)
        assert decoded['instruction_export']['capture_policy'] == policy
        assert decoded['instruction_export']['selected_threads_complete']
        for process in processes:
            for native_row in process.launches:
                if native_row['coverage'] != 'pass_through':
                    process.decode_launch(native_row)
    assert len({p.contexts[r['context']]['uuid'] for p in processes for r in p.launches if 'raw' in r}) == 8
    coverage = decoded_read('coverage.json')
    selected = [r for r in coverage if r['coverage'] in {'instrumented', 'observed'}]
    assert len(selected) == sum(len(p.activities) for p in processes)
    configured = {image['id']: image for image in json.loads(
        Path(launch['native_runtime']['catalog']).read_text())['images']}
    excluded = [r for r in selected if r['coverage'] == 'observed']
    for process in processes:
        for row in process.launches:
            if row['coverage'] != 'pass_through' and row['image'] in configured:
                assert process.decode_launch(row)[1] == configured[row['image']]
    for row in excluded:
        assert row['image'] in configured and configured[row['image']].get('probe_enabled') is False
        assert row['disposition'] == 'probe_excluded_original_image'
        assert row['intervals'] == 0 and row['raw'] is None
        assert row['instruction_export']['input_records'] == row['instruction_export']['exported_records'] == 0
    for row in coverage:
        if row['coverage'] == 'pass_through':
            assert not any(key in row.get('kernel', '').lower() for key in ('nccl', 'deep_ep'))
    summary, roles = [], ('dispatch_forward', 'combine_forward', 'dispatch_backward', 'combine_backward')
    reports = {rank: read(f'rank-{rank}.json') for rank in range(8)}
    for rank, report in reports.items():
        if mixed:
            assert {key: len(ranks) for key, ranks in report['parallel_groups'].items()} == {
                'DP': workload.dense_dp, 'TP': workload.tensor_parallel_size,
                'EP': workload.expert_parallel_size, 'expert_DP': workload.expert_dp}
            assert all(rank in ranks for ranks in report['parallel_groups'].values())
            assert all(call['successful'] for call in report['collective_calls'])
        process = next(p for p in processes if p.identity.rank == rank)
        assert report['status'] == 'passed' and report['session'] == launch['session']
        assert report['pid'] == process.identity.pid and report['gpu_uuid'] == launch['devices'][rank]['uuid']
        assert [r['step'] for r in report['steps']] == list(range(1, workload.train_iters + 1))
        recorded = [r for r in report['steps'] if r['capture_selected']]
        assert [r['step'] for r in recorded] == steps
        for row in recorded:
            assert row['optimizer_updated'] and row['forward_backward_executions'] == 1
            if policy is not None:
                assert row['timing_clock'] == 'CLOCK_MONOTONIC_RAW'
                assert (row['scope_begin_host_ns'] <= row['training_begin_host_ns'] <
                        row['training_end_host_ns'] <= row['scope_end_host_ns'])
                assert row['elapsed_ns'] == row['training_end_host_ns'] - row['training_begin_host_ns']
            assert row['losses'] and all(math.isfinite(v) for v in [*row['losses'].values(), row['grad_norm']])
            assert all(row['deepep_calls'][role] == workload.num_layers *
                       workload.manifest()['gradient_accumulation_steps'] for role in roles)
            assert row['nccl_python_calls']['dense_reduce_scatter'] > 0
            assert row['nccl_python_calls']['dense_all_gather'] > 0
            kernels = [r for r in selected if r['rank'] == rank and r['step'] == row['step']]
            counts = collections.Counter(r['library'] for r in kernels)
            assert counts['NCCL'] > 0 and counts['DeepEP'] > 0
            assert any('AllGather' in r['kernel'] for r in kernels if r['library'] == 'NCCL')
            assert any('ReduceScatter' in r['kernel'] for r in kernels if r['library'] == 'NCCL')
            assert any('dispatch' in r['kernel'] for r in kernels if r['library'] == 'DeepEP')
            assert any('combine' in r['kernel'] for r in kernels if r['library'] == 'DeepEP')
            summary.append({'rank': rank, 'step': row['step'], 'kernel_counts': dict(counts),
                            'losses': row['losses'], 'optimizer_updated': True})
    trace = normalize_chrome_document(decoded_read('communication.json'))
    if mixed:
        trace = annotate_parallel_roles(trace, selected, reports)
        for rank in range(8):
            for step in steps:
                categories = {e.get('cat') for e in trace['traceEvents']
                              if e.get('args', {}).get('rank') == rank and e['args'].get('step') == step}
                assert {'DP.NCCL.kernel', 'TP.NCCL.kernel', 'EP.DeepEP.kernel'} <= categories, (rank, step, categories)
                dense = [e for e in trace['traceEvents'] if e.get('cat') == 'DP.NCCL.kernel'
                         and e['args']['rank'] == rank and e['args']['step'] == step
                         and len(e['args']['role_evidence']['group_ranks']) == workload.dense_dp]
                assert any('ReduceScatter' in e['name'] for e in dense), (rank, step, 'dense gradient reduction')
                assert any('AllGather' in e['name'] for e in dense), (rank, step, 'dense parameter gathering')
    assert all(type(r['pid']) is int and type(r['tid']) is int for r in trace['traceEvents'])
    trace_path = decoded_directory / 'communication.chrome.json'
    kernels = [r for r in trace['traceEvents'] if r.get('cat', '').endswith('.kernel')]
    assert len(kernels) == len(selected)
    assert {r['args']['launch_id'] for r in kernels} == {r['launch_id'] for r in selected}
    assert all(r['dur'] > 0 and r['ts'] >= 0 for r in kernels)
    if not mixed:
        assert {r['cat'] for r in kernels} == {'DP.NCCL.kernel', 'EP.DeepEP.kernel'}
    from neutrino.sys_sync.runtime import implementation_identity
    build = json.loads(Path(launch['native_runtime']['build']).read_text())
    assert build['implementation'] == implementation_identity()
    assert hashlib.sha256(Path(build['library']).read_bytes()).hexdigest() == build['sha256']
    assert hashlib.sha256(Path(launch['native_runtime']['catalog']).read_bytes()).hexdigest() == launch['native_runtime']['catalog_sha256']
    modification_audit = {}
    loop_mode = decoded['interval_mode'] == 'loop-boundary'
    if mixed and not loop_mode:
        previous_path = directory.parent / 'train-ten-activity-1/launch.json'
        previous = json.loads(previous_path.read_text())
        previous_build = json.loads(Path(previous['native_runtime']['build']).read_text())
        assert build['implementation'].keys() == previous_build['implementation'].keys()
        historical_differences = sorted(name for name in build['implementation']
                                        if build['implementation'][name] != previous_build['implementation'][name])
        assert historical_differences == ['sys_sync/export.py']
        assert build['sha256'] == previous_build['sha256']
        assert launch['native_runtime']['catalog_sha256'] == previous['native_runtime']['catalog_sha256']
        assert launch['sources'] == previous['sources']
        current_harness = launch['environment_identity']['harness_files']
        previous_harness = previous['environment_identity']['harness_files']
        assert current_harness.keys() == previous_harness.keys()
        for name, checksum in current_harness.items():
            assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == checksum
        changed = sorted(name for name in current_harness if current_harness[name] != previous_harness[name])
        assert changed == sorted('neutrino/examples/moe_training/' + name for name in
                                 ('config.py', 'gpu_fixture.py', 'launch.py', 'preflight.py', 'worker.py'))
        modification_audit = {
            'reference_capture': str(previous_path.parent), 'changed_harness_files': changed,
            'updated_test_files': ['neutrino/tests/test_unified_training.py', 'neutrino/tests/test_training_evidence.py'],
            'unchanged_harness_files': sorted(set(current_harness) - set(changed)),
            'tracing_implementation_matches_reused_build': True,
            'reused_build': launch['native_runtime']['build'],
            'historical_source_differences': historical_differences,
            'historical_difference_note': 'The exporter was updated before this mixed request; the entire current implementation matches the preexisting runtime-activity-final build',
            'native_binary_unchanged_from_previous_capture': True,
            'nccl_catalog_unchanged': True, 'upstream_revisions_unchanged': True,
            'role_annotation': 'Application-side host collective audit plus offline process-group labels; no native communicator attribution'}
    if loop_mode:
        previous_path = ROOT.parent / 'artifacts/unified-sys/train-long-trace-1/launch.json'
        previous = json.loads(previous_path.read_text())
        previous_build = json.loads(Path(previous['native_runtime']['build']).read_text())
        assert launch['workload'] == previous['workload']
        assert launch['sources'] == previous['sources']
        if policy is None:
            assert launch['environment_identity'] == previous['environment_identity']
        else:
            for key in ('libraries', 'deep_ep_extension_sha256', 'dataset_helper'):
                assert launch['environment_identity'][key] == previous['environment_identity'][key]
            for name, checksum in launch['environment_identity']['harness_files'].items():
                assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == checksum
        catalog = json.loads(Path(launch['native_runtime']['catalog']).read_text())
        from neutrino.sys_sync.probe import loop_selection
        from neutrino.sys_sync.runtime import catalog_loop_depth
        assert catalog['interval_mode'] == 'loop-boundary'
        depths = {library: catalog_loop_depth(catalog, library) for library in ('NCCL', 'DeepEP')}
        depth = next(iter(depths.values())) if len(set(depths.values())) == 1 else None
        assert decoded.get('loop_depth', 1) == depth
        assert decoded.get('loop_depths', dict.fromkeys(depths, depth)) == depths
        assert all(image['interval_mode'] == 'loop-boundary' and
                   image.get('loop_depth', 1) == image['probe'].get('loop_depth', 1) == depths[image['label']] and
                   image['probe']['loop_selection'] == loop_selection(depths[image['label']])
                   for image in catalog['images'])
        assert decoded['counts']['NCCL_loop_intervals'] > 0 and decoded['counts']['DeepEP_loop_intervals'] > 0
        modification_audit = {
            'reference_capture': str(previous_path.parent), 'workload_unchanged': True,
            'harness_unchanged': launch['environment_identity'] == previous['environment_identity'],
            'upstream_revisions_unchanged': True,
            'native_binary_unchanged_from_previous_capture': build['sha256'] == previous_build['sha256'],
            'historical_source_differences': sorted(name for name in build['implementation']
                if build['implementation'][name] != previous_build['implementation'].get(name)),
            'current_implementation_matches_recorded_build': True,
            'interval_mode': 'loop-boundary', 'loop_depth': depth, 'loop_depths': depths,
            'loop_selection': loop_selection(depth) if depth is not None else 'per_library',
            'synchronization_success': 'unclassified'}
    raw_records = sum(r['intervals'] for r in selected)
    with trace_path.open('x') as output:
        json.dump(trace, output, separators=(',', ':'))
        output.write('\n')
    result = {'status': 'passed', 'session': launch['session'],
              'topology': f'DP{workload.dense_dp}/TP{workload.tensor_parallel_size}/EP{workload.expert_parallel_size}',
              'expert_dp': workload.expert_dp, 'tracing_core_matches_reused_build': not loop_mode,
              'tracing_core_matches_recorded_build': True, 'interval_mode': decoded['interval_mode'],
              'loop_depth': decoded.get('loop_depth', 1), 'loop_depths': decoded.get('loop_depths', {}),
              'modification_audit': modification_audit,
              'capture_policy': policy,
              'selected_threads_complete': policy is not None,
              'excluded_original_image_launches': len(excluded),
              'instrumented_launches': len(selected) - len(excluded),
              'all_threads_complete': not excluded and all(r.get('capture_sampling', 'all') == 'all' for r in selected),
              'runtime_build': launch['native_runtime']['build'], 'runtime_sha256': build['sha256'],
              'parallel_role_kernel_counts': dict(collections.Counter(r['cat'] for r in kernels)),
              'captured_steps': steps, 'validated_rank_steps': len(summary),
              'kernel_spans': len(kernels), 'instruction_records': raw_records if not loop_mode else None,
              'raw_interval_records': raw_records,
              'kernel_counts': dict(collections.Counter(r['library'] for r in selected)),
              'launch_apis': dict(collections.Counter(r['api'] for r in selected)),
              'all_optimizer_updates_verified': True, 'raw_record_overflows': 0,
              'cupti_dropped_records': 0, 'cross_gpu_reference': 'CLOCK_MONOTONIC_RAW',
              'maximum_estimated_instruction_alignment_uncertainty_ns': decoded['maximum_estimated_uncertainty_ns'],
              'communication_trace': str(trace_path),
              'communication_trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              'communication_trace_bytes': trace_path.stat().st_size,
              'rank_steps': summary}
    destination = directory / 'acceptance.json'
    with destination.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rank_steps'}, indent=2))
    return result


def duration_statistics(values):
    import math
    import statistics
    values = sorted(values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Expected nonempty finite nonnegative durations')
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'p90': values[math.ceil(len(values) * 0.9) - 1], 'min': values[0], 'max': values[-1],
            'sum': sum(values)}


def communication_statistics(events):
    from collections import defaultdict
    groups, steps = defaultdict(list), defaultdict(float)
    for event in events:
        if not event.get('cat', '').endswith('.kernel'):
            continue
        args = event['args']
        role = args.get('parallel_role', event['cat'].split('.')[0])
        if 'EXPERT_DATA_PARALLEL' in args.get('role_evidence', {}).get('group_description', ''):
            role = 'expert_DP'
        library = args['library']
        op = next((op for op in ('ReduceScatter', 'AllGather', 'AllReduce', 'Broadcast', 'dispatch', 'combine')
                   if op.lower() in event['name'].lower()), 'other')
        groups[f'{role}.{library}.{op}'].append(event['dur'])
        steps[role, args['rank'], args['step']] += event['dur'] / 1000
    per_role = defaultdict(list)
    for (role, rank, step), duration in steps.items():
        per_role[role].append(duration)
    return {'per_kernel_us': {key: duration_statistics(v) for key, v in sorted(groups.items())},
            'per_rank_step_kernel_sum_ms': {key: duration_statistics(v) for key, v in sorted(per_role.items())}}


def activity_kernel_event(event, rank, step):
    if event.get('cat') != 'kernel' or event.get('ph') != 'X':
        return None
    name = event['name'].lower()
    args = event.get('args', {})
    if 'deep_ep::ep::' in name:
        library, role = 'DeepEP', 'EP'
    elif name.startswith('nccl'):
        library = 'NCCL'
        description = args.get('Process Group Description', '')
        role = ('DP' if 'DATA_PARALLEL' in description else
                'TP' if 'TENSOR_MODEL_PARALLEL' in description else
                'Other' if description else 'Unattributed')
    else:
        return None
    return {**event, 'cat': f'{role}.{library}.kernel',
            'args': {**args, 'rank': rank, 'step': step, 'library': library, 'parallel_role': role,
                     'role_evidence': {'group_description': args.get('Process Group Description', ''),
                                       'method': 'PyTorch profiler collective metadata'}}}


def activity_control_statistics(directory):
    import json
    from collections import defaultdict
    directory = Path(directory)
    launch = json.loads((directory / 'launch.json').read_text())
    validation = json.loads((directory / 'validation.json').read_text())
    workload = Workload(**launch['workload']).validate()
    assert validation['status'] == 'passed' and not launch['trace'] and launch['profile_kernels']
    groups = defaultdict(list)
    communication = []
    total_kernels = 0
    for rank in range(workload.world_size):
        for step in range(workload.capture_first_step, workload.capture_last_step + 1):
            path = directory / f'activity-rank-{rank}-step-{step}.json'
            events = json.loads(path.read_text())['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
            assert kernels, (rank, step, 'missing GPU activities')
            total_kernels += len(kernels)
            for event in kernels:
                normalized = activity_kernel_event(event, rank, step)
                if normalized is None:
                    continue
                communication.append(normalized)
                if normalized['args']['library'] == 'NCCL':
                    op = next((op for op in ('ReduceScatter', 'AllGather', 'AllReduce', 'Broadcast')
                               if op.lower() in event['name'].lower()), 'other')
                    groups['NCCL.' + op].append(event['dur'])
    assert groups['NCCL.ReduceScatter'] and groups['NCCL.AllGather']
    return {'directory': str(directory.resolve()), 'gpu_kernel_count': total_kernels,
            'workload': workload.manifest(), 'instrumented': False,
            'per_kernel_us': {key: duration_statistics(v) for key, v in sorted(groups.items())},
            'role_statistics': communication_statistics(communication),
            'note': 'PyTorch/CUPTI activity-only control; roles use profiler collective metadata, not host durations'}


def compare_captures(reference, directory, *controls):
    import json
    reference, directory = Path(reference), Path(directory)
    summaries, workloads = [], []
    for path in (reference, directory):
        acceptance = json.loads((path / 'acceptance.json').read_text())
        assert acceptance['status'] == 'passed'
        launch = json.loads((path / 'launch.json').read_text())
        workloads.append(Workload(**launch['workload']).manifest())
        document = json.loads((path / 'decoded/communication.chrome.json').read_text())
        summaries.append(communication_statistics(document['traceEvents']))
    ratios = {key: summaries[1]['per_kernel_us'][key]['median'] / value['median']
              for key, value in summaries[0]['per_kernel_us'].items()
              if key in summaries[1]['per_kernel_us'] and value['median'] > 0}
    result = {'reference': str(reference.resolve()), 'current': str(directory.resolve()),
              'workloads': workloads, 'reference_statistics': summaries[0], 'current_statistics': summaries[1],
              'per_kernel_median_ratios': ratios,
              'activity_only_controls': [activity_control_statistics(path) for path in controls],
              'note': 'Instrumented GPU kernel durations, not host bars; sums are not critical-path time. '
                      'Larger durations do not alone establish a communication bottleneck.'}
    with (directory / 'communication-comparison.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2))
    return result


class CommunicationStatisticsTests(unittest.TestCase):
    def test_activity_library_and_group_identity(self):
        event = {'cat': 'kernel', 'ph': 'X', 'name': 'void deep_ep::ep::dispatch_impl(ncclDevComm)',
                 'dur': 10, 'args': {}}
        self.assertEqual(activity_kernel_event(event, 0, 3)['args']['library'], 'DeepEP')
        event['name'] = 'ncclDevKernel_AllGather_RING'
        event['args'] = {'Process Group Description': 'TENSOR_MODEL_PARALLEL_GROUP'}
        self.assertEqual(activity_kernel_event(event, 0, 3)['args']['parallel_role'], 'TP')
        event['args'] = {}
        self.assertEqual(activity_kernel_event(event, 0, 3)['args']['parallel_role'], 'Unattributed')
        event['name'] = 'some_compute_kernel'
        self.assertIsNone(activity_kernel_event(event, 0, 3))

    def test_duration_statistics(self):
        result = duration_statistics([30, 10, 20])
        self.assertEqual((result['count'], result['median'], result['p90'], result['sum']), (3, 20, 30, 60))
        for values in ([], [-1], [float('nan')]):
            with self.assertRaises(ValueError):
                duration_statistics(values)

    def test_roles_and_step_sums(self):
        def event(role, duration, group='DATA_PARALLEL_GROUP'):
            return {'cat': role + '.NCCL.kernel', 'name': 'NCCL AllGather', 'dur': duration,
                    'args': {'parallel_role': role, 'library': 'NCCL', 'rank': 0, 'step': 3,
                             'role_evidence': {'group_description': group}}}
        result = communication_statistics([event('DP', 100), event('DP', 200), event('TP', 50),
                                           event('DP', 400, 'EXPERT_DATA_PARALLEL_GROUP')])
        self.assertEqual(result['per_kernel_us']['DP.NCCL.AllGather']['median'], 150)
        self.assertAlmostEqual(result['per_rank_step_kernel_sum_ms']['DP']['median'], 0.3)
        self.assertEqual(result['per_kernel_us']['expert_DP.NCCL.AllGather']['count'], 1)


if __name__ == '__main__':
    if len(sys.argv) in (3, 4) and sys.argv[1] == '--validate-capture':
        validate_training_capture(*sys.argv[2:])
    elif len(sys.argv) >= 4 and sys.argv[1] == '--compare-captures':
        compare_captures(*sys.argv[2:])
    else:
        unittest.main()
