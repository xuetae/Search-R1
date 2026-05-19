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
from typing import Optional, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import LlamaConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .parallel_attention import ParallelLlamaAttention, ParallelLlamaAttentionRmPad
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .parallel_mlp import ParallelLlamaMLP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .parallel_rmsnorm import ParallelLlamaRMSNorm


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaDecoderLayer(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hidden_size = config.hidden_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.self_attn = ParallelLlamaAttention(config=config, megatron_config=megatron_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        residual = hidden_states

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.input_layernorm(hidden_states)

        # Note: sequence parallel is hidden inside ColumnParallelLinear
        # reduce scatter is hidden inside RowParallelLinear

        # Self Attention
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.self_attn(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states=hidden_states,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask=attention_mask,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids=position_ids,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # TODO: add sequence parallel operator reduce_scatter here

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = residual + hidden_states

        # Fully Connected
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        residual = hidden_states
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.post_attention_layernorm(hidden_states)

        # TODO: add sequence parallel operator all_gather here

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.mlp(hidden_states)

        # TODO: add sequence parallel operator reduce_scatter here

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = residual + hidden_states

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs = hidden_states

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return outputs


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaDecoderLayerRmPad(nn.Module):

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
        self.self_attn = ParallelLlamaAttentionRmPad(config=config, megatron_config=megatron_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.mlp = ParallelLlamaMLP(config, megatron_config=megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.input_layernorm = ParallelLlamaRMSNorm(config, megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.post_attention_layernorm = ParallelLlamaRMSNorm(config, megatron_config)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states: torch.Tensor,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sequence_length: int = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        indices: torch.Tensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cu_seqlens: int = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_seqlen_in_batch: int = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        residual = hidden_states  # (total_nnz // sp, 1, hidden_size)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # (total_nnz // sp, 1, hidden_size) -> all-gather (total_nnz, 1, hidden_size)
        # -> col + row -> reduce-scatter -> (total_nnz // sp, 1, hidden_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       position_ids=position_ids,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       sequence_length=sequence_length,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       indices=indices,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       cu_seqlens=cu_seqlens,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       max_seqlen_in_batch=max_seqlen_in_batch)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = residual + hidden_states

        # Fully Connected
        # shape changes same as attn
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        residual = hidden_states
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.mlp(hidden_states)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = residual + hidden_states

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs = hidden_states

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return outputs
