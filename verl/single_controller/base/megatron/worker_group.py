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
from typing import Dict

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .worker import DistRankInfo, DistGlobalInfo
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base import ResourcePool, WorkerGroup


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronWorkerGroup(WorkerGroup):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, resource_pool: ResourcePool, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(resource_pool=resource_pool, **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._megatron_rank_info = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._megatron_global_info: DistGlobalInfo = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_megatron(self, default_megatron_kwargs: Dict = None):
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError(f"MegatronWorkerGroup.init_megatron should be overwritten")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_megatron_rank_info(self, rank: int) -> DistRankInfo:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert 0 <= rank < self.world_size, f'rank must be from [0, world_size), Got {rank}'
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._megatron_rank_info[rank]

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def tp_size(self):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._megatron_global_info.tp_size

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def dp_size(self):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._megatron_global_info.dp_size

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pp_size(self):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._megatron_global_info is not None, "MegatronWorkerGroup._megatron_global_info must be initialized"
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._megatron_global_info.pp_size

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_megatron_global_info(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._megatron_global_info
