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
Adapted from Cruise.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Union

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
HALF_LIST = [16, "16", "fp16", "float16"]
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
FLOAT_LIST = [32, "32", "fp32", "float32"]
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
BFLOAT_LIST = ["bf16", "bfloat16"]


# 中文注释：下一行定义类，用于组织相关状态与行为。
class PrecisionType(object):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    HALF = "16"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    FLOAT = "32"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    FULL = "64"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    BFLOAT = "bf16"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MIXED = "mixed"

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def supported_type(precision: Union[str, int]) -> bool:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return any(x == precision for x in PrecisionType)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def supported_types() -> list[str]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return [x.value for x in PrecisionType]

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def is_fp16(precision):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return precision in HALF_LIST

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def is_fp32(precision):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return precision in FLOAT_LIST

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def is_bf16(precision):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return precision in BFLOAT_LIST

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def to_dtype(precision):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if precision in HALF_LIST:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return torch.float16
        # 中文注释：下一行继续判断其他条件分支。
        elif precision in FLOAT_LIST:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return torch.float32
        # 中文注释：下一行继续判断其他条件分支。
        elif precision in BFLOAT_LIST:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return torch.bfloat16
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise RuntimeError(f"unexpected precision: {precision}")

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def to_str(precision):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if precision == torch.float16:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 'fp16'
        # 中文注释：下一行继续判断其他条件分支。
        elif precision == torch.float32:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 'fp32'
        # 中文注释：下一行继续判断其他条件分支。
        elif precision == torch.bfloat16:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 'bf16'
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise RuntimeError(f"unexpected precision: {precision}")
