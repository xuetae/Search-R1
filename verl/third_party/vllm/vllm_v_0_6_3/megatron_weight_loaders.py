# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.linear import *
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.models import ModelRegistry


# NOTE(shengguangming): replace the origin weight loader function in the class
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Parallel Linear weight loader."""
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert (param.size() == loaded_weight.size(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )), "the parameter size is not align with the loaded weight size, param size: {}, loaded_weight size: {}".format(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param.size(), loaded_weight.size())
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert (param.data.dtype == loaded_weight.data.dtype
           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
           ), "if we want to shared weights, the data type should also be the same"

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    param.data = loaded_weight.data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Default weight loader."""
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert param.size() == loaded_weight.size()
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert (param.data.dtype == loaded_weight.data.dtype
           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
           ), "if we want to shared weights, the data type should also be the same"

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    param.data = loaded_weight.data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gpt2_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "lm_head.weight" in name:
            # GPT-2 ties the weights of the embedding layer and the final
            # linear layer.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ".attn.bias" in name or ".attn.masked_bias" in name:
            # Skip attention mask.
            # NOTE: "c_attn.bias" should not be skipped.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not name.startswith("transformer."):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = "transformer." + name
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param = params_dict[name]
        # The HF's GPT-2 implementation uses Conv1D instead of Linear.
        # Because of this, we need to transpose the weights.
        # Note(zhuohan): the logic below might break quantized models.
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if conv1d_weight_name not in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not name.endswith(".weight"):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # TODO: check megatron
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            loaded_weight = loaded_weight.t()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_loader(param, loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # NOTE(shengguangming): the megatron llama may have this prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("embedding.word_embeddings", "model.embed_tokens"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_proj", "self_attn.o_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc2", "mlp.down_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("decoder.final_layernorm", "model.norm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("output_layer", "lm_head"),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = _replace_name(name, params_mapping)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if name.endswith(".bias") and name not in params_dict:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("embedding.word_embeddings", "model.embed_tokens"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_proj", "self_attn.o_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "input_layernorm",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "input_layernorm",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc2", "mlp.down_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("decoder.final_layernorm", "model.norm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("output_layer", "lm_head"),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = _replace_name(name, params_mapping)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if name.endswith(".bias") and name not in params_dict:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _replace_name(megatron_name, name_mapping):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for m_name, v_name in name_mapping:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if m_name not in megatron_name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "layers" in megatron_name:  # deal with decoder layers
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            megatron_name = megatron_name.replace("decoder", "model")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            megatron_name_list = megatron_name.split(".")
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list = megatron_name_list[:3]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list.append(v_name)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name = ".".join(param_name_list)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list = megatron_name_list[:3]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_or_bias = megatron_name_list[-1]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list.append(v_name)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list.append(weight_or_bias)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name = ".".join(param_name_list)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return param_name
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_name = megatron_name.replace(m_name, v_name)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return param_name


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_megatron_core_te_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("embedding.word_embeddings", "model.embed_tokens"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_proj", "self_attn.o_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc2", "mlp.down_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("decoder.final_layernorm", "model.norm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("output_layer", "lm_head"),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = _replace_name(name, params_mapping)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if name.endswith(".bias") and name not in params_dict:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_megatron_core_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("embedding.word_embeddings", "model.embed_tokens"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("self_attention.linear_proj", "self_attn.o_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "input_layernorm",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "input_layernorm",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("mlp.linear_fc2", "mlp.down_proj"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("decoder.final_layernorm", "model.norm"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("output_layer", "lm_head"),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # NOTE(shengguangming): the megatron llama may have this prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = _replace_name(name, params_mapping)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if name.endswith(".bias") and name not in params_dict:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _replace_name(megatron_name, name_mapping):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for m_name, v_name in name_mapping:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if m_name not in megatron_name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "layers" in megatron_name:  # deal with decoder layers
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            megatron_name = megatron_name.replace("decoder", "model")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            megatron_name_list = megatron_name.split(".")
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list = megatron_name_list[:3]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list.append(v_name)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name = ".".join(param_name_list)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list = megatron_name_list[:3]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_or_bias = megatron_name_list[-1]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list.append(v_name)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name_list.append(weight_or_bias)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param_name = ".".join(param_name_list)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return param_name
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_name = megatron_name.replace(m_name, v_name)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return param_name


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def mistral_megatron_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # TODO: need to implement a general way to deal with prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight)


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__ = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ColumnParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MergedColumnParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    QKVParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    RowParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    VocabParallelEmbedding: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ParallelLMHead: parallel_weight_loader,
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): latest commit in vllm fix awq for this function and add load_weights
    # "default_weight_loader": default_weight_loader
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}

# for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
#     # setattr(layer_class, 'megatron_weight_loader', weight_loader)
#     layer_class.weight_loader = weight_loader

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__ = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "GPT2LMHeadModel": gpt2_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "LlamaForCausalLM": llama_megatron_weight_loader,  # use te backend for open-source megatron
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "LLaMAForCausalLM": llama_megatron_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "MistralForCausalLM": mistral_megatron_weight_loader,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}


# the actor model is .state_dict()
# Load megatron weights
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_megatron_weights(actor_weights: Dict, vllm_model: nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_model = vllm_model.cuda()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_model_weight_loader(arch: str):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if arch in __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return __MODEL_MEGATRON_WEIGHT_LOADER_REGISTRY__[arch]
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_megatron_weight_loader():
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for layer_class, weight_loader in __LAYER_WEIGHT_MEGATRON_LOADER_REGISTRY__.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        layer_class.weight_loader = weight_loader
