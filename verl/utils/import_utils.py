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
Utilities to check if packages are available.
We assume package availability won't change during runtime.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from functools import cache
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@cache
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_megatron_core_available():
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core import parallel_state as mpu
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return True
    # 中文注释：下一行处理异常分支，保证错误可控。
    except ImportError:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@cache
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_vllm_available():
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import vllm
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return True
    # 中文注释：下一行处理异常分支，保证错误可控。
    except ImportError:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def import_external_libs(external_libs=None):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if external_libs is None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not isinstance(external_libs, List):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        external_libs = [external_libs]
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import importlib
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for external_lib in external_libs:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        importlib.import_module(external_lib)
