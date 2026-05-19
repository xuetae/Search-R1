# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import math
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import tensor_parallel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import LlamaConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.models.llama.megatron.layers.parallel_linear import QKVParallelLinear

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import tensor_parallel as tp_utils


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LlamaRotaryEmbedding(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dim = dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_position_embeddings = max_position_embeddings
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.base = base
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inv_freq = 1.0 / (self.base**(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                device=self.inv_freq.device,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                dtype=torch.get_default_dtype())

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_seq_len_cached = seq_len
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        emb = torch.cat((freqs, freqs), dim=-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if seq_len > self.max_seq_len_cached:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scaling_factor = scaling_factor
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(dim, max_position_embeddings, base, device)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_seq_len_cached = seq_len
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        t = t / self.scaling_factor

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        emb = torch.cat((freqs, freqs), dim=-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scaling_factor = scaling_factor
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(dim, max_position_embeddings, base, device)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_seq_len_cached = seq_len

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if seq_len > self.max_position_embeddings:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            base = self.base * ((self.scaling_factor * seq_len / self.max_position_embeddings) -
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                (self.scaling_factor - 1))**(self.dim / (self.dim - 2))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inv_freq = 1.0 / (base**(torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        emb = torch.cat((freqs, freqs), dim=-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def rotate_half(x):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Rotates half the hidden dims of the input."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    x1 = x[..., :x.shape[-1] // 2]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    x2 = x[..., x.shape[-1] // 2:]
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.cat((-x2, x1), dim=-1)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return q_embed, k_embed


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if n_rep == 1:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return hidden_states
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaAttention(nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hidden_size = config.hidden_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_heads = config.num_attention_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.head_dim = self.hidden_size // self.num_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_key_value_heads = config.num_key_value_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_position_embeddings = config.max_position_embeddings
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rope_theta = config.rope_theta

        # assign values after tp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.num_heads % tp_size == 0, f'num_head must be divisible by tp_size. Got num_head={self.num_heads}, tp_size={tp_size}'
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.num_key_value_heads % tp_size == 0, \
            f'num_key_value_heads must be divisible by tp_size. Got num_key_value_heads={self.num_key_value_heads}, tp_size={tp_size}'

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_heads_per_tp = self.num_heads // tp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_key_value_heads_per_tp = self.num_key_value_heads // tp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hidden_size_per_tp = self.hidden_size // tp_size

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (self.head_dim * self.num_heads) != self.hidden_size:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f" and `num_heads`: {self.num_heads}).")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert row_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)

        # [self.q_size, self.k_size, self.v_size]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.qkv_proj = QKVParallelLinear(input_size=self.hidden_size,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          num_heads=self.num_heads,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          num_key_value_heads=self.num_key_value_heads,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          head_dim=self.head_dim,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          bias=config.attention_bias,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          gather_output=False,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          skip_bias_add=False,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          **column_kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.q_size = self.num_heads_per_tp * self.head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.k_size = self.num_key_value_heads_per_tp * self.head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.v_size = self.num_key_value_heads_per_tp * self.head_dim

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.o_proj = tensor_parallel.RowParallelLinear(input_size=self.num_heads * self.head_dim,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        output_size=self.hidden_size,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        bias=config.attention_bias,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        input_is_parallel=True,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        skip_bias_add=False,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        **row_kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_rope()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_rope(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.rope_scaling is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.rotary_emb = LlamaRotaryEmbedding(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.head_dim,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_position_embeddings=self.max_position_embeddings,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                base=self.rope_theta,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scaling_type = self.config.rope_scaling["type"]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scaling_factor = self.config.rope_scaling["factor"]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if scaling_type == "linear":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.head_dim,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    max_position_embeddings=self.max_position_embeddings,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    scaling_factor=scaling_factor,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    base=self.rope_theta,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
            # 中文注释：下一行继续判断其他条件分支。
            elif scaling_type == "dynamic":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.head_dim,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    max_position_embeddings=self.max_position_embeddings,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    scaling_factor=scaling_factor,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    base=self.rope_theta,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states: torch.Tensor,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        bsz, q_len, _ = hidden_states.size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        qkv = self.qkv_proj(hidden_states)[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states = query_states.view(bsz, q_len, self.num_heads_per_tp, self.head_dim).transpose(1, 2)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_tp, self.head_dim).transpose(1, 2)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_seq_len = key_states.shape[-2]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if attn_weights.size() != (bsz, self.num_heads_per_tp, q_len, kv_seq_len):
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"Attention weights should be of size {(bsz, self.num_heads_per_tp, q_len, kv_seq_len)}, but is"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f" {attn_weights.size()}")

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if attention_mask is not None:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output = torch.matmul(attn_weights, value_states)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if attn_output.size() != (bsz, self.num_heads_per_tp, q_len, self.head_dim):
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"`attn_output` should be of size {(bsz, self.num_heads_per_tp, q_len, self.head_dim)}, but is"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f" {attn_output.size()}")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_tp)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output = self.o_proj(attn_output)[0]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return attn_output


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Remove padding Attention
- Using Flash-attn 2
- Compatible with sequence parallel
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.utils import is_flash_attn_2_available
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn.functional as F

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from einops import rearrange

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if is_flash_attn_2_available():
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from flash_attn import flash_attn_varlen_func
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_rotary_pos_emb_rmpad(q, k, cos, sin, position_ids, indices, sequence_length):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size = position_ids.shape[0]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    q = pad_input(q, indices, batch_size, sequence_length)  # (batch_size, seqlen, num_head, head_dim)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    k = pad_input(k, indices, batch_size, sequence_length)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    q_embed = index_first_axis(rearrange(q_embed, "b s ... -> (b s) ..."), indices)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    k_embed = index_first_axis(rearrange(k_embed, "b s ... -> (b s) ..."), indices)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return q_embed, k_embed


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from flash_attn.layers.rotary import apply_rotary_emb


# use flash-attn rotary embeddings with rmpad
# cos/sin shoudl be: (seq_length, rotary_dim / 2)
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_rotary_pos_emb_rmpad_flash(q, k, cos, sin, cu_seqlens, max_seqlen):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    q_embed = apply_rotary_emb(q,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               cos,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               sin,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               interleaved=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               inplace=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               cu_seqlens=cu_seqlens,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               max_seqlen=max_seqlen)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    k_embed = apply_rotary_emb(k,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               cos,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               sin,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               interleaved=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               inplace=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               cu_seqlens=cu_seqlens,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               max_seqlen=max_seqlen)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return q_embed, k_embed


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaAttentionRmPad(ParallelLlamaAttention):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_states: torch.Tensor,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids: Optional[torch.LongTensor] = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sequence_length: int = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                indices: torch.Tensor = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                cu_seqlens: torch.Tensor = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_seqlen_in_batch: int = None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_nnz, _, _ = hidden_states.size()  # This is the total_nnz padded after sequence parallel

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_nnz = total_nnz * mpu.get_tensor_model_parallel_world_size()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        qkv = self.qkv_proj(hidden_states)[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size],
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           dim=-1)  # (total_nnz, 1, hidden_size)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sequence_parallel_pad = total_nnz - cu_seqlens[-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_nnz = cu_seqlens[-1]  # total_nnz before sp padding
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_states = query_states[:total_nnz]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            key_states = key_states[:total_nnz]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            value_states = value_states[:total_nnz]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dime x hidden_dim
        # therefore we just need to keep the original shape
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states = query_states.view(total_nnz, self.num_heads_per_tp, self.head_dim)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states = key_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        value_states = value_states.view(total_nnz, self.num_key_value_heads_per_tp, self.head_dim)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cos, sin = self.rotary_emb(value_states, seq_len=sequence_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]  # flash attn only needs half
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states, key_states = apply_rotary_pos_emb_rmpad_flash(query_states,
                                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                    key_states,
                                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                    cos,
                                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                    sin,
                                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                    cu_seqlens=cu_seqlens,
                                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                    max_seqlen=max_seqlen_in_batch)
        # query_states, key_states = apply_rotary_pos_emb_rmpad(query_states, key_states, cos, sin, position_ids, indices,

        # TODO: llama does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_dtype = query_states.dtype
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if input_dtype == torch.float32:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_states = query_states.to(torch.float16)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            key_states = key_states.to(torch.float16)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            value_states = value_states.to(torch.float16)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output_unpad = flash_attn_varlen_func(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_states,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            key_states,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            value_states,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cu_seqlens_q=cu_seqlens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cu_seqlens_k=cu_seqlens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_seqlen_q=max_seqlen_in_batch,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_seqlen_k=max_seqlen_in_batch,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dropout_p=dropout_rate,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            softmax_scale=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            causal=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output_unpad = attn_output_unpad.to(input_dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output_unpad = attn_output_unpad.reshape(total_nnz, 1, self.hidden_size_per_tp).contiguous()

        # sequence parallel reduce_scatter is performed inside RowColumnParallel if enabled
        # Here we need to repad
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attn_output_unpad = F.pad(attn_output_unpad, pad=(0, 0, 0, 0, 0, sequence_parallel_pad))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output_unpad = self.o_proj(attn_output_unpad)[0]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return attn_output_unpad
