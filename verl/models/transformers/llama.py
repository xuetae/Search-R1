# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional, List, Union, Tuple, Unpack, Callable

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.cache_utils import Cache
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.utils import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.modeling_flash_attention_utils import _flash_attention_forward
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads, get_ulysses_sequence_parallel_world_size

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = logging.get_logger(__name__)

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_flash_attn_forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states: torch.Tensor,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.LongTensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        past_key_value: Optional[Cache] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_attentions: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_cache: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_position: Optional[torch.LongTensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **kwargs,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
        adapt from transformers 4.47.1
        """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_attentions = False

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bsz, q_len, _ = hidden_states.size()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    query_states = self.q_proj(hidden_states)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    key_states = self.k_proj(hidden_states)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # trade off: repeat first and then all to all
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    ########## AlltoAll for Ulysses ##########
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if ulysses_sp_size > 1:
        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    full_q_len = query_states.size(2)  # full seq length

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if position_embeddings is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.warning_once(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "removed and `position_embeddings` will be mandatory.")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cos, sin = self.rotary_emb(value_states, position_ids)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cos, sin = position_embeddings
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    query_states = query_states.transpose(1, 2)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    key_states = key_states.transpose(1, 2)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    value_states = value_states.transpose(1, 2)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_dtype = query_states.dtype
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if input_dtype == torch.float32:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.is_autocast_enabled():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        # 中文注释：下一行继续判断其他条件分支。
        elif hasattr(self.config, "_pre_quantization_dtype"):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_dtype = self.config._pre_quantization_dtype
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_dtype = self.q_proj.weight.dtype

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.warning_once(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            f" {target_dtype}.")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states = query_states.to(target_dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states = key_states.to(target_dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        value_states = value_states.to(target_dtype)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attn_output = _flash_attention_forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        query_states,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        key_states,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        value_states,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        full_q_len,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids=position_ids,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dropout=dropout_rate,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sliding_window=getattr(self, "sliding_window", None),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_causal=self.is_causal,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **kwargs,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()
    ########## AlltoAll for Ulysses ##########
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if ulysses_sp_size > 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attn_output = self.o_proj(attn_output)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not output_attentions:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_weights = None

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return attn_output, attn_weights, past_key_value
