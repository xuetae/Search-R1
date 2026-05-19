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
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_hf_weight_loader():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print("no hf weight loader need to be updated")
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(actor_weights, Dict)
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):  # TODO
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_weights.keys():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            del actor_weights["lm_head.weight"]
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
