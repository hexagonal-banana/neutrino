"""Application-side adapter for pinned Megatron and unmodified DeepEP 2.5.

Installation changes only this process's Megatron callables, never source files
or DeepEP's API. Import and install before constructing the model.
"""
import inspect
from pathlib import Path
import subprocess

MEGATRON_REVISION = 'd7157825675a65c2503a061ffd1fffa7886df9b7'


def install(megatron_root):
    import torch
    import deep_ep
    from megatron.core.transformer.moe import fused_a2a, token_dispatcher

    root = Path(megatron_root).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != MEGATRON_REVISION or not Path(fused_a2a.__file__).resolve().is_relative_to(root):
        raise RuntimeError('The adapter requires the isolated, pinned Megatron checkout')
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Megatron source must remain unmodified')
    if deep_ep.__version__.split('+')[0] != '2.5.0':
        raise RuntimeError('The adapter requires DeepEP 2.5.0')
    required = {'do_expand', 'do_cpu_sync', 'defer_epilogue', 'handle'}
    if not required.issubset(inspect.signature(deep_ep.EPBuffer.dispatch).parameters):
        raise RuntimeError('Incompatible DeepEP dispatch ABI')
    if getattr(fused_a2a, '_neutrino_adapter', None) is not None:
        return fused_a2a._neutrino_adapter
    counts = dict.fromkeys(('dispatch_forward', 'combine_forward', 'dispatch_backward',
                            'combine_backward'), 0)
    buffers = {}

    def get_buffer(group, num_max_tokens_per_rank, hidden, num_topk):
        key = (group, hidden, num_topk)
        old = buffers.get(key)
        if old is not None:
            if old.num_max_tokens_per_rank < num_max_tokens_per_rank:
                raise RuntimeError('Workload token capacity changed after buffer creation')
            return old
        # Establish the communicator before DeepEP asks PyTorch for its pointer.
        torch.distributed.all_reduce(torch.zeros(1, device='cuda'), group=group)
        buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=num_max_tokens_per_rank,
                                  hidden=hidden, num_topk=num_topk, deterministic=True,
                                  allow_hybrid_mode=False, explicitly_destroy=True)
        buffers[key] = buffer
        return buffer

    class Dispatch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, buffer, x, indices, weights, experts, capacity, alignment, sms,
                    async_finish, allocate_on_comm_stream):
            counts['dispatch_forward'] += 1
            indices = indices.to(dtype=deep_ep.topk_idx_t).contiguous()
            weights = weights.float().contiguous()
            x = x.contiguous()
            event = buffer.capture() if async_finish and allocate_on_comm_stream else None
            rx, ri, rw, handle, done = buffer.dispatch(
                x, topk_idx=indices, topk_weights=weights, num_experts=experts,
                num_max_tokens_per_rank=capacity, expert_alignment=alignment, num_sms=sms,
                previous_event=event, async_with_compute_stream=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
                do_expand=False, do_cpu_sync=True, defer_epilogue=False)
            if async_finish:
                done.current_stream_wait()
            ctx.buffer, ctx.handle = buffer, handle
            ctx.sms, ctx.async_finish, ctx.allocate = handle.num_sms, async_finish, allocate_on_comm_stream
            ctx.recv_layout = (rx.shape, rx.dtype, rx.device)
            ctx.set_materialize_grads(False)
            return rx, ri, rw, torch.tensor(handle.num_recv_tokens_per_expert_list), handle

        @staticmethod
        def backward(ctx, grad_x, grad_indices, grad_weights, grad_counts, grad_handle):
            counts['dispatch_backward'] += 1
            if grad_x is None:
                shape, dtype, device = ctx.recv_layout
                grad_x = torch.zeros(shape, dtype=dtype, device=device)
            grad_x = grad_x.contiguous()
            grad_weights = None if grad_weights is None else grad_weights.float().contiguous()
            event = ctx.buffer.capture() if ctx.async_finish and ctx.allocate else None
            gx, gw, done = ctx.buffer.combine(
                grad_x, handle=ctx.handle, topk_weights=grad_weights, num_sms=ctx.sms,
                previous_event=event, async_with_compute_stream=ctx.async_finish,
                allocate_on_comm_stream=ctx.allocate, defer_epilogue=False)
            if ctx.async_finish:
                done.current_stream_wait()
            return None, gx, None, gw, None, None, None, None, None, None

    class Combine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, buffer, x, handle, sms, async_finish, allocate_on_comm_stream):
            counts['combine_forward'] += 1
            x = x.contiguous()
            event = buffer.capture() if async_finish and allocate_on_comm_stream else None
            result, weights, done = buffer.combine(
                x, handle=handle, num_sms=sms, previous_event=event,
                async_with_compute_stream=async_finish, allocate_on_comm_stream=allocate_on_comm_stream,
                defer_epilogue=False)
            if async_finish:
                done.current_stream_wait()
            ctx.buffer, ctx.handle = buffer, handle
            ctx.sms = handle.num_sms if sms == 0 else sms
            ctx.async_finish, ctx.allocate = async_finish, allocate_on_comm_stream
            ctx.set_materialize_grads(False)
            return result, weights

        @staticmethod
        def backward(ctx, grad_x, grad_weights):
            counts['combine_backward'] += 1
            if grad_weights is not None:
                raise RuntimeError('Unweighted combine must not receive a weight gradient')
            grad_x = grad_x.contiguous()
            event = ctx.buffer.capture() if ctx.async_finish and ctx.allocate else None
            result, _, _, _, done = ctx.buffer.dispatch(
                grad_x, handle=ctx.handle, num_sms=ctx.sms, previous_event=event,
                async_with_compute_stream=ctx.async_finish, allocate_on_comm_stream=ctx.allocate,
                do_expand=False, do_cpu_sync=False, defer_epilogue=False)
            if ctx.async_finish:
                done.current_stream_wait()
            return None, result, None, None, None, None

    def dispatch(buffer, x, token_indices, token_probs, num_experts, num_max_tokens_per_rank,
                 expert_alignment=1, num_sms=0, async_finish=False, allocate_on_comm_stream=False):
        return Dispatch.apply(buffer, x, token_indices, token_probs, num_experts,
                              num_max_tokens_per_rank, expert_alignment, num_sms,
                              async_finish, allocate_on_comm_stream)

    def combine(buffer, x, handle, num_sms=0, async_finish=False, allocate_on_comm_stream=False):
        return Combine.apply(buffer, x, handle, num_sms, async_finish, allocate_on_comm_stream)

    def close():
        for buffer in buffers.values():
            buffer.destroy()
        buffers.clear()

    for module in (fused_a2a, token_dispatcher):
        module.get_elastic_buffer = get_buffer
        module.deepepv2_dispatch = dispatch
        module.deepepv2_combine = combine
    result = {'counts': counts, 'close': close, 'dispatch': dispatch, 'combine': combine,
              'get_buffer': get_buffer}
    fused_a2a._neutrino_adapter = result
    return result
