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
from typing import Dict, Union, Optional, Iterable, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_hf_weight_loader():
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from vllm.model_executor.models.gemma import GemmaForCausalLM
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    GemmaForCausalLM.load_weights = gemma_load_weights


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gemma_load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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
    params_dict = dict(self.named_parameters())
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    loaded_params = set()
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, loaded_weight in weights:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for (param_name, shard_name, shard_id) in stacked_params_mapping:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if shard_name not in name:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = name.replace(shard_name, param_name)
            # Skip loading extra bias for GPTQ models.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name.endswith(".bias") and name not in params_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params_dict[name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader = param.weight_loader
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight_loader(param, loaded_weight, shard_id)
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
                norm_weight = loaded_weight + 1.0  # prevent inplace modify actor weights
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param = params_dict[name]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader(param, norm_weight)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param = params_dict[name]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight_loader(param, loaded_weight)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loaded_params.add(name)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    unloaded_params = params_dict.keys() - loaded_params
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if unloaded_params:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise RuntimeError("Some weights are not initialized from checkpoints: "
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           f"{unloaded_params}")


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(actor_weights, Dict)
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):  # TODO
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vllm_model.load_weights(actor_weights.items())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for _, module in vllm_model.named_modules():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        quant_method = getattr(module, "quant_method", None)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if quant_method is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            quant_method.process_weights_after_loading(module)
        # FIXME: Remove this after Mixtral is updated
        # to use quant_method.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hasattr(module, "process_weights_after_loading"):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            module.process_weights_after_loading()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_model = vllm_model.cuda()
