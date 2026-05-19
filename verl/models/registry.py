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
import importlib
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Optional, Type

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn

# Supported models using HF Rmpad
# TODO(sgm): HF may supported more than listed here, we should add more after testing
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_REOVEPAD_MODELS = {'llama': LlamaConfig, 'mistral': MistralConfig, 'gemma': GemmaConfig, 'qwen2': Qwen2Config}


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def check_model_support_rmpad(model_type: str):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(model_type, str)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not model_type in _REOVEPAD_MODELS.keys():
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError(f"Model architecture {model_type} is not supported for now. "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         f"RMPad supported architectures: {_REOVEPAD_MODELS.keys()}."
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         f"Please set `use_remove_padding=False` in the model config.")


# Supported models in Megatron-LM
# Architecture -> (module, class).
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_MODELS = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "LlamaForCausalLM":
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ("llama", ("ParallelLlamaForCausalLMRmPadPP", "ParallelLlamaForValueRmPadPP", "ParallelLlamaForCausalLMRmPad")),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "MistralForCausalLM": ("mistral", ("ParallelMistralForCausalLMRmPadPP", "ParallelMistralForValueRmPadPP",
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       "ParallelMistralForCausalLMRmPad"))
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}


# return model class
# 中文注释：下一行定义类，用于组织相关状态与行为。
class ModelRegistry:

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model_cls(model_arch: str, value=False) -> Optional[Type[nn.Module]]:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_arch not in _MODELS:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron = "megatron"

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        module_name, model_cls_name = _MODELS[model_arch]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not value:  # actor/ref
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_cls_name = model_cls_name[0]
        # 中文注释：下一行继续判断其他条件分支。
        elif value:  # critic/rm
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_cls_name = model_cls_name[1]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        module = importlib.import_module(f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return getattr(module, model_cls_name, None)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_supported_archs() -> List[str]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return list(_MODELS.keys())
