"""CPU-only DP/TP/EP workload specification; defaults remain DP8/EP8/TP1."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Workload:
    world_size: int = 8
    expert_parallel_size: int = 8
    tensor_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    num_layers: int = 2
    hidden_size: int = 1536
    num_attention_heads: int = 12
    ffn_hidden_size: int = 4096
    num_experts: int = 8
    topk: int = 2
    seq_length: int = 256
    micro_batch_size: int = 1
    global_batch_size: int = 8
    train_iters: int = 12
    capture_first_step: int = 3
    capture_last_step: int = 12
    seed: int = 1234
    learning_rate: float = 0.0001

    @classmethod
    def from_file(cls, path):
        import json
        from pathlib import Path
        return cls(**json.loads(Path(path).read_text())).validate()

    def validate(self):
        for key, value in asdict(self).items():
            if key != 'learning_rate' and (type(value) is not int or value <= 0):
                raise ValueError(f'Expected positive integer: {key}')
        if self.world_size not in (2, 8):
            raise ValueError('This workload supports two or eight GPUs')
        if self.world_size % self.tensor_parallel_size or self.world_size % self.expert_parallel_size:
            raise ValueError('TP and EP must divide world size')
        if self.expert_tensor_parallel_size != 1:
            raise ValueError('Expert tensor parallelism is outside this workload configuration')
        if self.num_attention_heads % self.tensor_parallel_size or self.seq_length % self.tensor_parallel_size:
            raise ValueError('Heads and sequence length must divide into TP shards')
        if self.num_experts % self.expert_parallel_size or self.topk > self.num_experts:
            raise ValueError('Invalid expert routing dimensions')
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('Hidden size must be divisible by head count')
        if self.global_batch_size % (self.dense_dp * self.micro_batch_size):
            raise ValueError('Global batch must divide into dense DP microbatches')
        if not 1 <= self.capture_first_step <= self.capture_last_step <= self.train_iters:
            raise ValueError('Capture must select complete training steps')
        if not 0 < self.learning_rate < 1:
            raise ValueError('Invalid learning rate')
        return self

    @property
    def dense_dp(self):
        return self.world_size // self.tensor_parallel_size

    @property
    def expert_dp(self):
        return self.world_size // (self.expert_parallel_size * self.expert_tensor_parallel_size)

    def manifest(self):
        self.validate()
        return {**asdict(self), 'dense_dp': self.dense_dp, 'expert_dp': self.expert_dp,
                'pipeline_parallel_size': 1, 'context_parallel_size': 1,
                'gradient_accumulation_steps': self.global_batch_size //
                (self.dense_dp * self.micro_batch_size),
                'communication_roles': {'nccl': ['dense_gradient_reduce_scatter',
                                                'dense_parameter_all_gather'] +
                                               (['tensor_parallel_collectives'] if self.tensor_parallel_size > 1 else []) +
                                               (['expert_data_parallel_collectives'] if self.expert_dp > 1 else []),
                                        'deepep': ['dispatch_forward', 'combine_forward',
                                                   'dispatch_backward', 'combine_backward']}}

    def megatron_args(self):
        self.validate()
        values = {
            'num-layers': self.num_layers, 'hidden-size': self.hidden_size,
            'num-attention-heads': self.num_attention_heads,
            'ffn-hidden-size': self.ffn_hidden_size, 'moe-ffn-hidden-size': self.ffn_hidden_size,
            'num-experts': self.num_experts, 'expert-model-parallel-size': self.expert_parallel_size,
            'expert-tensor-parallel-size': self.expert_tensor_parallel_size,
            'tensor-model-parallel-size': self.tensor_parallel_size,
            'pipeline-model-parallel-size': 1, 'context-parallel-size': 1,
            'moe-router-topk': self.topk, 'moe-router-load-balancing-type': 'aux_loss',
            'moe-aux-loss-coeff': 0.01, 'moe-router-dtype': 'fp32',
            'moe-token-dispatcher-type': 'flex', 'moe-flex-dispatcher-backend': 'deepepv2',
            'seq-length': self.seq_length, 'max-position-embeddings': self.seq_length,
            'micro-batch-size': self.micro_batch_size, 'global-batch-size': self.global_batch_size,
            'train-iters': self.train_iters, 'lr': self.learning_rate,
            'min-lr': self.learning_rate / 10, 'lr-warmup-iters': 1,
            'lr-decay-iters': self.train_iters, 'lr-decay-style': 'cosine',
            'weight-decay': 0.1, 'clip-grad': 1.0, 'optimizer': 'adam',
            'normalization': 'RMSNorm', 'position-embedding-type': 'rope',
            'attention-dropout': 0.0, 'hidden-dropout': 0.0,
            'transformer-impl': 'transformer_engine', 'seed': self.seed,
            'num-workers': 0, 'log-interval': 1, 'eval-iters': 0,
            'eval-interval': self.train_iters, 'distributed-timeout-minutes': 10,
            'tokenizer-type': 'NullTokenizer', 'vocab-size': 50257,
            'rerun-mode': 'disabled',
        }
        flags = ['bf16', 'swiglu', 'disable-bias-linear', 'untie-embeddings-and-output-weights',
                 'use-distributed-optimizer', 'moe-grouped-gemm', 'moe-per-layer-logging',
                 'no-masked-softmax-fusion', 'no-gradient-accumulation-fusion', 'mock-data']
        if self.tensor_parallel_size > 1:
            flags.append('sequence-parallel')
        return [arg for key, value in values.items() for arg in (f'--{key}', str(value))] + [
            f'--{flag}' for flag in flags]
