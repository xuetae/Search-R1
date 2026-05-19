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
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""PyTorch LLaMA model with Megatron-style acceleration."""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional, Tuple, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.utils.checkpoint
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import tensor_parallel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.modeling_outputs import BaseModelOutputWithPast
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.models.llama.configuration_llama import LlamaConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import sequence_parallel as sp_utils
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import tensor_parallel as tp_utils
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .layers import ParallelLlamaDecoderLayer, ParallelLlamaRMSNorm, ParallelLlamaDecoderLayerRmPad
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
TODO: 
1. Add weight initialization. Here we need to be careful on TP weight init.
2. Add sequence parallel
3. Load checkpoint from meta LLama pretrained checkpoint
"""


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bsz, tgt_len = input_ids_shape
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask_cond = torch.arange(mask.size(-1), device=device)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask = mask.to(dtype)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bsz, src_len = mask.size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tgt_len = tgt_len if tgt_len is not None else src_len

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    inverted_mask = 1.0 - expanded_mask

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaModel(nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.padding_idx = config.pad_token_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vocab_size = config.vocab_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                   embedding_dim=config.hidden_size,
                                                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                   **embedding_kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.layers = nn.ModuleList(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            [ParallelLlamaDecoderLayer(config, megatron_config) for _ in range(config.num_hidden_layers)])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        combined_attention_mask = None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if input_shape[-1] > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            combined_attention_mask = _make_causal_mask(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                inputs_embeds.dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=inputs_embeds.device,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              tgt_len=input_shape[-1]).to(inputs_embeds.device)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       combined_attention_mask)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return combined_attention_mask

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids: torch.LongTensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """

        Args:
            input_ids: input ids. shape (batch_size, seq_length)
            attention_mask: attention_mask. shape (batch_size, seq_length)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size, seq_length = input_ids.shape
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = inputs_embeds

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx, decoder_layer in enumerate(self.layers):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_outputs = decoder_layer(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_states,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                attention_mask=attention_mask,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids=position_ids,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states = layer_outputs

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.norm(hidden_states)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return hidden_states


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaForCausalLM(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = ParallelLlamaModel(config, megatron_config=megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vocab_size = config.vocab_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=config.hidden_size,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            output_size=config.vocab_size,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            bias=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            gather_output=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            skip_bias_add=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            **column_kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids: torch.LongTensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs = self.model(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids=input_ids,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask=attention_mask,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids=position_ids,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = outputs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self.lm_head(hidden_states)[0]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.float()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return CausalLMOutputWithPast(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            loss=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits=logits,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            past_key_values=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attentions=None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaModelRmPad(nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.padding_idx = config.pad_token_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vocab_size = config.vocab_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                   embedding_dim=config.hidden_size,
                                                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                   **embedding_kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.layers = nn.ModuleList(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            [ParallelLlamaDecoderLayerRmPad(config, megatron_config) for _ in range(config.num_hidden_layers)])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.norm = ParallelLlamaRMSNorm(config, megatron_config)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids: torch.Tensor,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids: Optional[torch.LongTensor] = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sequence_length: int = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                indices: torch.Tensor = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                cu_seqlens: int = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_seqlen_in_batch: int = None) -> Union[Tuple, BaseModelOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

        # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs_embeds = inputs_embeds.transpose(0, 1)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = inputs_embeds
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx, decoder_layer in enumerate(self.layers):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_outputs = decoder_layer(hidden_states,
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
            hidden_states = layer_outputs

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = self.norm(hidden_states)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return hidden_states


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaForCausalLMRmPad(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = ParallelLlamaModelRmPad(config, megatron_config=megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vocab_size = config.vocab_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_head()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_head(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.config.hidden_size,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            output_size=self.config.vocab_size,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            bias=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            gather_output=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            skip_bias_add=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            **column_kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self.lm_head(hidden_states)[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)  # (total_nnz_padded, 1, vocab_size)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return logits

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids: torch.LongTensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size, sequence_length = input_ids.shape

        # remove padding here
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                              attention_mask)  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = sp_utils.pad_to_sequence_parallel(input_ids)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = input_ids.transpose(0, 1)  # (1, total_nnz+pad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs = self.model(input_ids=input_ids,
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
        hidden_states = outputs

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self._forward_head(hidden_states)

        # remove padding from sequence parallel
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            totol_nnz = cu_seqlens[-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = logits[:totol_nnz]  # (total_nnz_padded)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension
        # add removed padding back
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = pad_input(logits, indices, batch_size,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return CausalLMOutputWithPast(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            loss=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits=logits,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            past_key_values=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attentions=None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaForValueRmPad(ParallelLlamaForCausalLMRmPad):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_head(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_head(self, hidden_states):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.float()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return logits

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids: torch.LongTensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = super().forward(input_ids, attention_mask, position_ids)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output.logits = torch.squeeze(output.logits, dim=-1)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Support pipeline parallelism
"""


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaModelRmPadPP(nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    This model definition supports pipeline parallelism. To support pp and vpp,
    - This model only contains layer in this pp stage and vpp chunk
    - When calling get_model in Megatron, this rank will instantiate all the vpp chunks in this pp.
    Args:
        config: LlamaConfig
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, pre_process, post_process):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.padding_idx = config.pad_token_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vocab_size = config.vocab_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pre_process = pre_process
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.post_process = post_process
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        embedding_kwargs = tp_utils.get_default_kwargs_for_parallel_embedding()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert embedding_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(embedding_kwargs, self.megatron_config)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if pre_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.embed_tokens = tensor_parallel.VocabParallelEmbedding(num_embeddings=config.vocab_size,
                                                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                       embedding_dim=config.hidden_size,
                                                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                       **embedding_kwargs)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.embed_tokens = None

        # pp_rank = megatron_config.pipeline_model_parallel_rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pp_size = megatron_config.pipeline_model_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_layer_per_pp = config.num_hidden_layers // pp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vpp_size = megatron_config.virtual_pipeline_model_parallel_size

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if vpp_size is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.num_layer_vpp_chunk = self.num_layer_per_pp // vpp_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.num_layer_this_model = self.num_layer_vpp_chunk
            # vpp_rank = megatron_config.virtual_pipeline_model_parallel_rank
            # self.offset = vpp_rank * (
            #         config.num_hidden_layers // megatron_config.virtual_pipeline_model_parallel_size) + \
            #             (megatron_config.pipeline_model_parallel_rank * self.num_layer_vpp_chunk)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.num_layer_this_model = self.num_layer_per_pp
            # self.offset = pp_rank * self.num_layer_per_pp

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        layers = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(self.num_layer_this_model):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer = ParallelLlamaDecoderLayerRmPad(config, megatron_config)
            # setattr(layer, 'hidden_layer_index', self.offset + i)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layers.append(layer)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.layers = nn.ModuleList(layers)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if post_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.norm = ParallelLlamaRMSNorm(config, megatron_config)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.norm = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def set_input_tensor(self, input_tensor):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.input_tensor = input_tensor

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids: torch.Tensor,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids: Optional[torch.LongTensor] = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sequence_length: int = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                indices: torch.Tensor = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                cu_seqlens: int = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_seqlen_in_batch: int = None) -> Union[Tuple, BaseModelOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """

        Args:
            input_ids: input ids. shape (1, totol_nnz)
            position_ids: position ids. shape (batch_size, seq_length)

        Returns:

        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.pre_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inputs_embeds = self.embed_tokens(input_ids)  # (1, total_nnz) -> (1, total_nnz, hidden_size)

            # vocab parallel embedding will not do sequence parallel reduce-scatter in open source megatron
            # so need to deal with it by handle here:
            # (1, total_nnz, hidden_size) -> (total_nnz, 1, hidden_size) -> (total_nnz // sp, 1, hidden_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inputs_embeds = inputs_embeds.transpose(0, 1)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.megatron_config.sequence_parallel:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states = inputs_embeds
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # self.hidden_states should be passed by Megatron
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states = self.input_tensor

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx, decoder_layer in enumerate(self.layers):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_outputs = decoder_layer(hidden_states,
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
            hidden_states = layer_outputs

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.post_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states = self.norm(hidden_states)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return hidden_states


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaForCausalLMRmPadPP(nn.Module):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: LlamaConfig, megatron_config: ModelParallelConfig, pre_process, post_process):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = ParallelLlamaModelRmPadPP(config,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               megatron_config=megatron_config,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               pre_process=pre_process,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               post_process=post_process)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.share_embeddings_and_output_weights = None  # workaround, megatron requires this attr
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vocab_size = config.vocab_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pre_process = pre_process
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.post_process = post_process
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if post_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._init_head()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def set_input_tensor(self, input_tensor):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(input_tensor) == 1
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model.set_input_tensor(input_tensor[0])

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_head(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=self.config.hidden_size,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            output_size=self.config.vocab_size,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            bias=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            gather_output=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            skip_bias_add=False,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            **column_kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_head(self, hidden_states):
        # all_gather from sequence parallel region is performed inside lm_head
        # logits shape before forward_head hidden_states.shape: [4, 32, 4096]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self.lm_head(hidden_states)[0]
        # logits shape after forward_head logits.shape: [8, 32, 8]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.float()  # (total_nnz_padded, 1, vocab_size // tp)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return logits

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # original input
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        *,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids: torch.LongTensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        ```"""

        # Note that input_ids, attention_mask and position_ids should be passed to every pp layer.
        # In the first pp, input_ids will be used, in other pp layers hidden_states will be used inside self.model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size, sequence_length = input_ids.shape
        # remove padding here
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                                    attention_mask)  # (total_nnz, 1)

        # pad input_ids to multiple of tp for all tp ranks
        # TODO: for better performance, the sp padding should be removed at each layer. Not sure the performance gap
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz+pad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs = self.model(input_ids=input_ids_rmpad,
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

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.post_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states = outputs
            # print(f'hidden_states.shape = {hidden_states.shape}') # torch.Size([4, 32, 4096])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = self._forward_head(hidden_states)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = torch.squeeze(logits, dim=1)  # remove the artificial batch dimension # torch.Size([8, 32, 16])

            # remove padding from sequence parallel
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.megatron_config.sequence_parallel:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                totol_nnz = cu_seqlens[-1]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits = logits[:totol_nnz]  # (total_nnz_padded)
            # add removed padding back. If input is already rmpad, we let the caller pad_input
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = pad_input(logits, indices, batch_size,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               seqlen=sequence_length)  # (batch_size, sequence_length, vocab_size)

            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return CausalLMOutputWithPast(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                loss=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits=logits,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                past_key_values=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_states=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                attentions=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return outputs


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelLlamaForValueRmPadPP(ParallelLlamaForCausalLMRmPadPP):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_head(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tp_utils.update_kwargs_with_config(column_kwargs, self.megatron_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lm_head = nn.Linear(in_features=self.config.hidden_size, out_features=1, bias=False)
        # lm_head is effectively the same as sequence parallel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sp_utils.mark_parameter_as_sequence_parallel(self.lm_head.weight)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_head(self, hidden_states):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self.lm_head(hidden_states)  # (total_nnz_padded // tp, 1, 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.float()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.megatron_config.sequence_parallel:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return logits

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        *,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids: torch.LongTensor = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask: Optional[torch.Tensor] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids: Optional[torch.LongTensor] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.post_process:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.logits = torch.squeeze(output.logits, dim=-1)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return output
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return output
