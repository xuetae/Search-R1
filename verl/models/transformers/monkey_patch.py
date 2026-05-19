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
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Apply monkey-patch function to models
"""

#### Open Source Models
#### transformers version < 4.48


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_monkey_patch_to_llama():
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.models.transformers.llama import llama_flash_attn_forward
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    LlamaFlashAttention2.forward = llama_flash_attn_forward


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_monkey_patch_to_qwen2():
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.models.transformers.qwen2 import qwen2_flash_attn_forward
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    Qwen2FlashAttention2.forward = qwen2_flash_attn_forward


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_PATCH_NAME_TO_FUNC = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'llama': apply_monkey_patch_to_llama,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'qwen2': apply_monkey_patch_to_qwen2,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_monkey_patch(config: PretrainedConfig, verbose=True):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not is_transformers_version_in_range("4.45.0", "4.47.1"):
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise AssertionError("The installed `transformers` version doesn't support ulysses patch. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "Please install a version between 4.45.0 and 4.47.1 to use this ulysses feature.")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    success_apply_monkey_patch = False
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.model_type in _PATCH_NAME_TO_FUNC:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _PATCH_NAME_TO_FUNC[config.model_type]()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        success_apply_monkey_patch = True

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if success_apply_monkey_patch and verbose:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Applying monkey patch to model {config.model_type}')
    # 中文注释：下一行继续判断其他条件分支。
    elif not success_apply_monkey_patch:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError(f'Ulysses for model {config.model_type} is not implemented, \
                                   please set `ulysses_sequence_parallel_size=1`')

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return success_apply_monkey_patch


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from functools import lru_cache
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from packaging import version
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import importlib.metadata


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@lru_cache()
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_transformers_version_in_range(min_version: str, max_version: str) -> bool:
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # Get the installed version of the transformers library
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformers_version = importlib.metadata.version("transformers")
    # 中文注释：下一行处理异常分支，保证错误可控。
    except importlib.metadata.PackageNotFoundError:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ModuleNotFoundError("The `transformers` package is not installed.")

    # Check if the version is within the specified range
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return version.parse(min_version) <= version.parse(transformers_version) <= version.parse(max_version)
