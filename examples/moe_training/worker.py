"""Internal worker: GPU imports are forbidden until the launcher guard is present."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def collective_role(description):
    if 'DATA_PARALLEL' in description:
        return 'DP'
    if 'TENSOR_MODEL_PARALLEL' in description:
        return 'TP'
    return 'Other'


def install_collective_audit(report):
    """Record application group evidence without changing CUDA hooks or launches."""
    import functools
    import inspect
    import threading
    import time
    import torch.distributed as dist

    state = {'step': None}
    local = threading.local()
    report['collective_calls'] = []

    def wrap(original, *, context_manager=False):
        if getattr(original, '_neutrino_collective_audit', False):
            return original
        signature = inspect.signature(original)

        @functools.wraps(original)
        def call(*args, **kwargs):
            if state['step'] is None or getattr(local, 'active', False):
                return original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            group = bound.arguments.get('group')
            group = dist.group.WORLD if group is None else group
            if dist.get_backend(group) != 'nccl':
                return original(*args, **kwargs)
            description = str(getattr(group, 'group_desc', 'unknown'))
            row = {'step': state['step'], 'api': original.__name__,
                   'group_name': str(group.group_name), 'group_description': description,
                   'group_ranks': dist.get_process_group_ranks(group),
                   'role': collective_role(description),
                   'async_op': bool(bound.arguments.get('async_op', bound.arguments.get('async_ops', False))),
                   'host_thread': threading.get_native_id(), 'successful': False,
                   'submission_phase': 'context_exit' if context_manager else 'function_call'}

            def submit(operation):
                local.active = True
                row['begin_host_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                try:
                    result = operation()
                    row['successful'] = True
                    return result
                finally:
                    row['end_host_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    local.active = False
                    report['collective_calls'].append(row)

            if context_manager:
                from contextlib import contextmanager

                @contextmanager
                def scope():
                    manager = original(*args, **kwargs)
                    entered = manager.__enter__()
                    try:
                        yield entered
                    except BaseException:
                        exception = sys.exc_info()
                        if not submit(lambda: manager.__exit__(*exception)):
                            raise
                    else:
                        submit(lambda: manager.__exit__(None, None, None))

                return scope()
            return submit(lambda: original(*args, **kwargs))

        call._neutrino_collective_audit = True
        return call

    for name in ('all_reduce', 'all_gather', 'all_gather_into_tensor', '_all_gather_base',
                 'reduce_scatter', 'reduce_scatter_tensor', '_reduce_scatter_base',
                 'broadcast', 'all_to_all_single'):
        setattr(dist, name, wrap(getattr(dist, name)))
    dist._coalescing_manager = wrap(dist._coalescing_manager, context_manager=True)
    return state, wrap


def main():
    if (os.environ.get('UNIFIED_GUARDED_LAUNCH') != '1' or 'LOCAL_RANK' not in os.environ or
            Path(sys.prefix).resolve() != (ROOT / '.venv-unified').resolve()):
        raise SystemExit('Use examples/moe_training/launch.py with confirmed allocation')
    import contextlib
    import inspect
    import json
    import math
    import runpy
    import time
    from config import Workload
    from preflight import MEGATRON, digest, harness_files, runtime_libraries
    from setup_environment import verify_sources
    from adapter import install

    run_dir = Path(os.environ['UNIFIED_RUN_DIR']).resolve()
    launch = json.loads((run_dir / 'launch.json').read_text())
    workload = Workload(**launch['workload'])
    workload.validate()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != workload.world_size:
        raise RuntimeError('Unexpected worker count')
    sys.path.insert(0, str(ROOT / 'neutrino'))
    sys.path.insert(0, str(MEGATRON))
    report = {'schema': 1, 'status': 'failed', 'rank': rank, 'pid': os.getpid(),
              'session': os.environ['NEUTRINO_SESSION_ID'],
              'gpu_uuid': launch['devices'][local_rank]['uuid'], 'steps': [],
              'kernel_activity_independently_verified': False}
    api = None
    torch = None
    try:
        if launch['trace']:
            from neutrino.capture import capture_range, require_runtime, finalize
            require_runtime()
        import torch
        import deep_ep
        report['sources'] = verify_sources()
        deep_ep.check_nccl_so()
        report['libraries'] = runtime_libraries()
        identity = launch['environment_identity']
        if (report['sources'] != launch['sources'] or report['libraries'] != identity['libraries'] or
                harness_files() != identity['harness_files'] or
                digest(deep_ep._C.__file__) != identity['deep_ep_extension_sha256']):
            raise RuntimeError('Worker environment changed after launch validation')
        from dataset_helper import install as install_dataset_helper
        report['dataset_helper'] = install_dataset_helper()
        if report['dataset_helper'] != identity['dataset_helper']:
            raise RuntimeError('Worker dataset helper differs from the preflight')
        torch.cuda.set_device(local_rank)
        api = install(MEGATRON)
        if launch['mode'] == 'fixture':
            from gpu_fixture import run
            report['fixture'] = run(api, trace=launch['trace'], workload=workload)
        else:
            from megatron.training import training
            from megatron.core.distributed import param_and_grad_buffer
            from megatron.core.rerun_state_machine import get_rerun_state_machine
            original_step = training.train_step
            signature = inspect.signature(original_step)
            collective_counts = {'dense_reduce_scatter': 0, 'dense_all_gather': 0}
            audit, audit_wrap = (install_collective_audit(report) if workload.tensor_parallel_size > 1
                                 else ({'step': None}, lambda function: function))
            if workload.tensor_parallel_size > 1:
                param_and_grad_buffer._coalescing_manager = audit_wrap(
                    param_and_grad_buffer._coalescing_manager, context_manager=True)

            def count_collective(original, key):
                original = audit_wrap(original)
                def call(*args, **kwargs):
                    group = kwargs.get('group')
                    if torch.distributed.get_world_size(group) == workload.dense_dp:
                        collective_counts[key] += 1
                    return original(*args, **kwargs)
                return call

            param_and_grad_buffer.dist_reduce_scatter_func = count_collective(
                param_and_grad_buffer.dist_reduce_scatter_func, 'dense_reduce_scatter')
            param_and_grad_buffer.dist_all_gather_func = count_collective(
                param_and_grad_buffer.dist_all_gather_func, 'dense_all_gather')

            def checked_step(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                step = bound.arguments['iteration'] + 1
                if step != len(report['steps']) + 1 or step > workload.train_iters:
                    raise RuntimeError('Unexpected or repeated training step')
                if get_rerun_state_machine().get_mode().value != 'disabled':
                    raise RuntimeError('Automatic training reruns are forbidden during capture')
                if 'parallel_groups' not in report:
                    from megatron.core import parallel_state as ps
                    groups = {'DP': ps.get_data_parallel_group(), 'TP': ps.get_tensor_model_parallel_group(),
                              'EP': ps.get_expert_model_parallel_group(), 'expert_DP': ps.get_expert_data_parallel_group()}
                    report['parallel_groups'] = {key: torch.distributed.get_process_group_ranks(pg)
                                                 for key, pg in groups.items()}
                    expected = {'DP': workload.dense_dp, 'TP': workload.tensor_parallel_size,
                                'EP': workload.expert_parallel_size, 'expert_DP': workload.expert_dp}
                    if any(len(report['parallel_groups'][key]) != size for key, size in expected.items()):
                        raise RuntimeError('Actual process groups differ from requested topology')
                optimizer = bound.arguments['optimizer']
                parameters = [p for p in optimizer.get_parameters() if p.numel()][:16]
                if not parameters:
                    raise RuntimeError('No optimizer shards available for update validation')
                before = [p.detach().reshape(-1)[:256].cpu().float().clone() for p in parameters]
                previous_calls = dict(api['counts'])
                previous_collectives = dict(collective_counts)
                forward_backward = bound.arguments['forward_backward_func']
                executions = 0

                def once(*a, **kw):
                    nonlocal executions
                    executions += 1
                    if executions != 1:
                        raise RuntimeError('Refusing repeated forward/backward schedule execution')
                    return forward_backward(*a, **kw)

                bound.arguments['forward_backward_func'] = once
                selected = workload.capture_first_step <= step <= workload.capture_last_step
                scope = capture_range('optimizer_step', step) if launch['trace'] and selected else contextlib.nullcontext()
                torch.cuda.synchronize()
                audit['step'] = step if selected else None
                profile = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                             torch.profiler.ProfilerActivity.CUDA])
                           if launch.get('profile_kernels') and selected else None)
                scope_begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                with profile if profile is not None else contextlib.nullcontext(), scope:
                    started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    result = original_step(*bound.args, **bound.kwargs)
                    torch.cuda.synchronize()
                    training_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                    elapsed = training_end - started
                    if executions != 1 or result[1] or result[3]:
                        raise RuntimeError('Training skipped an update or requested an early exit')
                scope_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                audit['step'] = None
                if profile is not None:
                    profile.export_chrome_trace(str(run_dir / f'activity-rank-{rank}-step-{step}.json'))
                losses = {key: float(value) for key, value in result[0].items()}
                grad_norm = float(result[5])
                if not losses or not all(math.isfinite(v) for v in [*losses.values(), grad_norm]):
                    raise RuntimeError('Nonfinite or absent training loss/gradient norm')
                after = [p.detach().reshape(-1)[:256].cpu().float() for p in parameters]
                if not all(torch.isfinite(p).all().item() for p in [*before, *after]):
                    raise RuntimeError('Nonfinite optimizer shard sample')
                changed = any(not torch.equal(a, b) for a, b in zip(before, after))
                calls = {key: value - previous_calls[key] for key, value in api['counts'].items()}
                collectives = {key: value - previous_collectives[key] for key, value in collective_counts.items()}
                if not all(calls.values()) or not all(collectives.values()):
                    raise RuntimeError(f'Missing step communication: {calls}, {collectives}')
                report['steps'].append({'step': step, 'capture_selected': selected, 'elapsed_ns': elapsed,
                                        'scope_begin_host_ns': scope_begin, 'scope_end_host_ns': scope_end,
                                        'training_begin_host_ns': started, 'training_end_host_ns': training_end,
                                        'timing_clock': 'CLOCK_MONOTONIC_RAW',
                                        'losses': losses, 'grad_norm': grad_norm, 'optimizer_updated': changed,
                                        'deepep_calls': calls, 'nccl_python_calls': collectives,
                                        'forward_backward_executions': executions})
                return result

            training.train_step = checked_step
            sys.argv = [str(MEGATRON / 'pretrain_gpt.py'), *workload.megatron_args()]
            try:
                runpy.run_path(sys.argv[0], run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
            if (len(report['steps']) != workload.train_iters or
                    not any(row['optimizer_updated'] for row in report['steps'] if row['capture_selected'])):
                raise RuntimeError('Expected completed optimizer steps and observed parameter updates')
            torch.cuda.synchronize()
            api['close']()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        if not all(api['counts'].values()):
            raise RuntimeError('Missing DeepEP forward/backward evidence')
        if launch['trace']:
            finalize()
        report['status'] = 'passed'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if api is not None:
            report['deepep_calls'] = dict(api['counts'])
        try:
            if torch is not None and torch.cuda.is_initialized():
                report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        except Exception as exc:
            report.update(status='failed', telemetry_error=f'{type(exc).__name__}: {exc}')
        (run_dir / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
