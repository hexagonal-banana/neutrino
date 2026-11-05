"""Guarded NCCL numeric and DeepEP autograd controls, before combined training."""
import os
from contextlib import nullcontext
from datetime import timedelta


def run(api, trace=False, workload=None):
    if os.environ.get('UNIFIED_GUARDED_LAUNCH') != '1':
        raise RuntimeError('Use the guarded workload launcher')
    import torch
    import torch.distributed as dist
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 8):
        raise RuntimeError('The fixture requires two or eight ranks')
    for peer in range(world):
        if peer != rank and not torch.cuda.can_device_access_peer(rank, peer):
            raise RuntimeError('Required GPU peer access is unavailable')
    group = dist.group.WORLD
    parallel = None
    group_checks = {}
    if workload is not None and (workload.tensor_parallel_size > 1 or workload.expert_parallel_size != world):
        from megatron.core import parallel_state as parallel
        parallel.initialize_model_parallel(tensor_model_parallel_size=workload.tensor_parallel_size,
                                           expert_model_parallel_size=workload.expert_parallel_size,
                                           expert_tensor_parallel_size=workload.expert_tensor_parallel_size)
        group = parallel.get_expert_model_parallel_group()
        for role, pg in (('DP', parallel.get_data_parallel_group()),
                         ('TP', parallel.get_tensor_model_parallel_group()),
                         ('expert_DP', parallel.get_expert_data_parallel_group())):
            size = dist.get_world_size(pg)
            value = torch.full((4096,), float(dist.get_rank(pg) + 1), device='cuda')
            dist.all_reduce(value, group=pg)
            torch.testing.assert_close(value, torch.full_like(value, size * (size + 1) / 2), rtol=0, atol=0)
            group_checks[role] = {'ranks': dist.get_process_group_ranks(pg), 'all_reduce': 'passed'}
        group_checks['EP'] = {'ranks': dist.get_process_group_ranks(group)}
    local_rank, group_size = dist.get_rank(group), dist.get_world_size(group)
    # Exercise every specialization before recording so compiler/rank skew is
    # excluded from the measurement; warmup still runs the full numeric oracle.
    warmup = numeric_checks(api, local_rank, group_size, group)
    torch.cuda.synchronize()
    dist.barrier()
    if trace:
        from neutrino.capture import capture_range
        scope = capture_range('communication_numeric_fixture', 1)
    else:
        scope = nullcontext()
    with scope:
        evidence = numeric_checks(api, local_rank, group_size, group)
        torch.cuda.synchronize()
    evidence['warmup'] = warmup
    evidence['groups'] = group_checks
    dist.barrier()
    api['close']()
    if parallel is not None:
        parallel.destroy_model_parallel()
    dist.destroy_process_group()
    return evidence


def numeric_checks(api, rank, world, group=None):
    import torch
    import torch.distributed as dist
    chunk = 262144
    base = torch.arange(world * chunk, device='cuda', dtype=torch.float32).remainder_(1024)
    reduced = torch.empty(chunk, device='cuda')
    dist.reduce_scatter_tensor(reduced, base + rank, group=group)
    expected = world * base[rank * chunk:(rank + 1) * chunk] + world * (world - 1) // 2
    torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
    gathered = torch.empty_like(base)
    dist.all_gather_into_tensor(gathered, reduced, group=group)
    torch.testing.assert_close(gathered, world * base + world * (world - 1) // 2, rtol=0, atol=0)
    evidence = {'nccl_reduce_scatter': 'passed', 'nccl_all_gather': 'passed',
                'nccl_elements_per_rank': chunk, 'deepep': []}
    n, hidden, experts, topk = 32, 1536, 8, 2
    local_experts = experts // world
    buffer = api['get_buffer'](group if group is not None else dist.group.WORLD, n, hidden, topk)
    for skew in (False, True):
        torch.manual_seed(321)
        all_x = (torch.randn(world * n, hidden, device='cuda') * 0.1).bfloat16()
        all_weights = (torch.rand(experts, hidden, device='cuda') + 0.5).bfloat16()
        router_value = torch.randn(hidden, experts, device='cuda') * 0.01
        x = all_x[rank * n:(rank + 1) * n].clone().requires_grad_()
        weights = all_weights[rank * local_experts:(rank + 1) * local_experts].clone().requires_grad_()
        router = router_value.clone().requires_grad_()
        bias = torch.zeros(experts, device='cuda')
        if skew:
            bias[:2] = 10
        scores, indices = (x.float() @ router + bias).topk(topk, dim=-1)
        probs = scores.softmax(-1)
        rx, ri, rp, counts, handle = api['dispatch'](buffer, x, indices, probs, experts, n,
                                                    async_finish=True)
        output = rx.float() * 0 + rp.sum() * 0 + weights.float().sum() * 0
        for expert in range(local_experts):
            coefficient = (rp * (ri == expert)).sum(-1, keepdim=True)
            output = output + rx.float() * weights[expert].float() * coefficient
        result, _ = api['combine'](buffer, output.bfloat16(), handle, async_finish=True)
        result.float().square().sum().div(n * hidden).backward()
        ref_x = all_x.clone().requires_grad_()
        ref_weights = all_weights.clone().requires_grad_()
        ref_router = router_value.clone().requires_grad_()
        ref_scores, ref_indices = (ref_x.float() @ ref_router + bias).topk(topk, dim=-1)
        ref_probs = ref_scores.softmax(-1)
        ref = ref_x.float() * 0 + ref_weights.float().sum() * 0
        for expert in range(experts):
            coefficient = (ref_probs * (ref_indices == expert)).sum(-1, keepdim=True)
            ref = ref + ref_x.float() * ref_weights[expert].float() * coefficient
        ref = ref.bfloat16()
        ref.float().square().sum().div(n * hidden).backward()
        dist.all_reduce(router.grad, group=group)
        errors = {}
        for name, actual, expected in (
                ('forward', result, ref[rank * n:(rank + 1) * n]),
                ('input_gradient', x.grad, ref_x.grad[rank * n:(rank + 1) * n]),
                ('expert_gradient', weights.grad,
                 ref_weights.grad[rank * local_experts:(rank + 1) * local_experts]),
                ('router_gradient', router.grad, ref_router.grad)):
            if not torch.isfinite(actual).all().item():
                raise RuntimeError(f'Nonfinite {name}')
            relative_error = ((actual.float() - expected.float()).norm() /
                              expected.float().norm().clamp_min(1e-8)).item()
            if relative_error > 0.04:
                raise RuntimeError(f'{name} relative L2 error {relative_error} exceeds 0.04')
            errors[name] = relative_error
        evidence['deepep'].append({'skew': skew, 'relative_l2': errors, 'counts': counts.tolist()})
    return evidence
