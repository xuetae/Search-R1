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
from verl.utils.import_utils import is_vllm_available, is_megatron_core_available

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .base import BaseShardingManager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .fsdp_ulysses import FSDPUlyssesShardingManager

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
AllGatherPPModel = None

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if is_megatron_core_available() and is_vllm_available():
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .megatron_vllm import AllGatherPPModel, MegatronVLLMShardingManager
# 中文注释：下一行继续判断其他条件分支。
elif AllGatherPPModel is not None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pass
# 中文注释：下一行处理前面条件都不满足时的默认分支。
else:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    AllGatherPPModel = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MegatronVLLMShardingManager = None

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if is_vllm_available():
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from .fsdp_vllm import FSDPVLLMShardingManager
# 中文注释：下一行处理前面条件都不满足时的默认分支。
else:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    FSDPVLLMShardingManager = None
