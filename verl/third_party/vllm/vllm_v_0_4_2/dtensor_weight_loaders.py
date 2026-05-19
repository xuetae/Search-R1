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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Iterable, Tuple
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed._tensor import DTensor, Shard, Replicate

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.linear import *
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.models import ModelRegistry
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "q_proj", "q"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "k_proj", "k"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "v_proj", "v"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("gate_up_proj", "gate_proj", 0),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("gate_up_proj", "up_proj", 1),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for (param_name, shard_name, shard_id) in stacked_params_mapping:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if shard_name not in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            stacked_name = name.replace(shard_name, param_name)
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[stacked_name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            break
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # lm_head is not used in vllm as it is tied with embed_token.
            # To prevent errors, skip loading lm_head.weight.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "lm_head.weight" in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name.endswith(".bias") and name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # GemmaRMSNorm is different from Llama's in that it multiplies
            # (1 + weight) to the output, instead of just weight.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "norm.weight" in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                norm_weight = local_loaded_weight + 1.0
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param = params_dict[name]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader(param, norm_weight.to(dtype=param.dtype))
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param = params_dict[name]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gptbigcode_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "lm_head.weight" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ".attn.bias" in name:
            # Skip attention mask.
            # NOTE: "c_attn.bias" should not be skipped.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param = params_dict[name]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def starcoder2_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "q_proj", "q"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "k_proj", "k"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "v_proj", "v"),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if weight_name not in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = name.replace(weight_name, param_name)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = param.weight_loader
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            break
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def llama_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (".qkv_proj", ".q_proj", "q"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (".qkv_proj", ".k_proj", "k"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (".qkv_proj", ".v_proj", "v"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (".gate_up_proj", ".gate_proj", 0),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (".gate_up_proj", ".up_proj", 1),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ("rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name):
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # With tie_word_embeddings, we can skip lm_head.weight
        # The weight might appear unnecessarily in the files if the model is
        # processed with quantization, LoRA, fine-tuning, etc.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if weight_name not in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name.endswith(".bias") and name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = param.weight_loader
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            break
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name.endswith(".bias") and name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def qwen2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "q_proj", "q"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "k_proj", "k"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("qkv_proj", "v_proj", "v"),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("gate_up_proj", "gate_proj", 0),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("gate_up_proj", "up_proj", 1),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "rotary_emb.inv_freq" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if weight_name not in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name.endswith(".bias") and name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = param.weight_loader
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            break
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name.endswith(".bias") and name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gpt2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pass


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def redistribute_dtensor(param_name: str, loaded_weights: DTensor, parallelize_plan: Dict = None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    param_name = _process_parameter_names(name=param_name)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if parallelize_plan is not None:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert param_name in parallelize_plan.keys(), \
            f"param name: {param_name} not in parallelize_plan :{parallelize_plan.keys()}"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        placement = parallelize_plan[param_name]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_loaded_weights = loaded_weights.redistribute(device_mesh=loaded_weights.device_mesh,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           placements=placement).to_local()
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_loaded_weights = loaded_weights.full_tensor()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return local_loaded_weights


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _process_parameter_names(name):
    # Remove '.weight' if it exists at the end of the string
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if name.endswith(".weight"):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = name[:-7]

    # Remove 'model.layers.x.' or 'model.' prefix
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "model.layers" in name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parts = name.split('.')
        # Reconstruct the string without 'model.layers.x.'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = '.'.join(parts[3:])  # parts[0] is 'model', parts[1] is 'layers', parts[2] is 'x'
    # 中文注释：下一行继续判断其他条件分支。
    elif name.startswith("model."):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = name[6:]  # Remove 'model.'

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return name


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__ = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'GPT2LMHeadModel': gpt2_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'LlamaForCausalLM': llama_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'LLaMAForCausalLM': llama_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'MistralForCausalLM': llama_dtensor_weight_loader,  # mistral is the same as llama in vLLM
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'InternLMForCausalLM': llama_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'AquilaModel': llama_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'AquilaForCausalLM': llama_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'Phi3ForCausalLM': llama_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'GemmaForCausalLM': gemma_dtensor_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'GPTBigCodeForCausalLM': gptbigcode_dtensor_load_weights,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'Starcoder2ForCausalLM': starcoder2_dtensor_load_weights,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'Qwen2ForCausalLM': qwen2_dtensor_weight_loader
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}


# the actor model is .state_dict()
# Load dtensor weights
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_dtensor_weights(actor_weights: Dict, vllm_model: nn.Module):
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
    if arch in __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__[arch]
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     f"Supported architectures: {__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__.keys()}")


# NOTE(sgm): we use per-parameter weight loader in each vllm sub
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_dtensor_weight_loader():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pass
