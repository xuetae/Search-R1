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
from typing import Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn


# NOTE(shengguangming): replace the origin weight loader function in the class
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def parallel_weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Parallel Linear weight loader."""
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert param.size() == loaded_weight.size(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ), 'the parameter size is not align with the loaded weight size, param size: {}, loaded_weight size: {}'.format(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param.size(), loaded_weight.size())
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    param.data = loaded_weight.data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Default weight loader."""
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert param.size() == loaded_weight.size()
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert param.data.dtype == loaded_weight.data.dtype, "if we want to shared weights, the data type should also be the same"

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
def llama_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # NOTE(shengguangming): the megatron llama may have this prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prefix = '0.module.module.'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if name[:len(prefix)] == prefix:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = name[len(prefix):]
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
def mistral_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    # TODO: need to implement a general way to deal with prefix
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prefix = '0.module.module.'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dict = dict(vllm_model.named_parameters())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in actor_weights.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if name[:len(prefix)] == prefix:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = name[len(prefix):]
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
