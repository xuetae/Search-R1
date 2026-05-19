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
from importlib.metadata import version, PackageNotFoundError


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_version(pkg):
    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return version(pkg)
    # 中文注释：下一行处理异常分支，保证错误可控。
    except PackageNotFoundError:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
package_name = 'vllm'
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
package_version = get_version(package_name)

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if package_version == '0.3.1':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_version = '0.3.1'
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_3_1.llm import LLM
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_3_1.llm import LLMEngine
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_3_1 import parallel_state
# 中文注释：下一行继续判断其他条件分支。
elif package_version == '0.4.2':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_version = '0.4.2'
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_4_2.llm import LLM
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_4_2.llm import LLMEngine
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_4_2 import parallel_state
# 中文注释：下一行继续判断其他条件分支。
elif package_version == '0.5.4':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_version = '0.5.4'
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_5_4.llm import LLM
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_5_4.llm import LLMEngine
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_5_4 import parallel_state
# 中文注释：下一行继续判断其他条件分支。
elif package_version == '0.6.3':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_version = '0.6.3'
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_6_3.llm import LLM
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_6_3.llm import LLMEngine
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .vllm_v_0_6_3 import parallel_state
# 中文注释：下一行处理前面条件都不满足时的默认分支。
else:
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'vllm version {package_version} not supported. Currently supported versions are 0.3.1, 0.4.2, 0.5.4 and 0.6.3.'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
