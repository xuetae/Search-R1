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
from typing import Dict, Optional

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .base import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.megatron.worker import DistRankInfo, DistGlobalInfo
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup


# NOTE(sgm): for opensource megatron-core
# 中文注释：下一行定义类，用于组织相关状态与行为。
class NVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name='get_megatron_rank_info')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._megatron_global_info: DistGlobalInfo = ray.get(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.execute_rank_zero_async(method_name='get_megatron_global_info'))


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 resource_pool: RayResourcePool,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 ray_cls_with_init: RayClassWithInitArgs,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 default_megatron_kwargs: Dict = None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(resource_pool=resource_pool,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         ray_cls_with_init=ray_cls_with_init,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         default_megatron_kwargs=default_megatron_kwargs,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.init_megatron(default_megatron_kwargs=default_megatron_kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name='get_megatron_rank_info')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._megatron_global_info: DistGlobalInfo = ray.get(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.execute_rank_zero_async(method_name='get_megatron_global_info'))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_megatron(self, default_megatron_kwargs: Optional[Dict] = None):
        # after super, we will call init of each worker
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self._is_init_with_detached_workers:
            # only init_megatron if the WorkerGroup is created from scratch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.execute_all_sync(method_name='init_megatron', default_megatron_kwargs=default_megatron_kwargs)
