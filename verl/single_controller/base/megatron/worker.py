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
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.worker import Worker, DistRankInfo, DistGlobalInfo


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronWorker(Worker):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, cuda_visible_devices=None) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(cuda_visible_devices)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_megatron_global_info(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core import parallel_state as mpu
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp_size = mpu.get_data_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        info = DistGlobalInfo(tp_size=tp_size, dp_size=dp_size, pp_size=pp_size)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return info

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_megatron_rank_info(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core import parallel_state as mpu
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp_rank = mpu.get_data_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        info = DistRankInfo(tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return info