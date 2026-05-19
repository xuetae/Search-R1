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
Contains a resharding manager that binds weights from FSDP zero3 to XPerfGPT
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .base import BaseShardingManager

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import random
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.device_mesh import DeviceMesh

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import allgather_dict_tensors
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.ulysses import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto


# 中文注释：下一行定义类，用于组织相关状态与行为。
class FSDPUlyssesShardingManager(BaseShardingManager):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Sharding manager to support data resharding when using FSDP + Ulysses
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, device_mesh: DeviceMesh):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_mesh = device_mesh
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.seed_offset = 12345

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __enter__(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # We have a global SP group
            # so we have to change to use model-specific sp group
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            set_ulysses_sequence_parallel_group(self.device_mesh['sp'].get_group())
            # TODO: check how to set seed for each model

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __exit__(self, exc_type, exc_value, traceback):
        # restore random states
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # revert to previous sp group
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            set_ulysses_sequence_parallel_group(self.prev_sp_group)
            # TODO: check how to set seed for each model

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        AllGather data from sp region
        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE
        In Ulysses, we need to make sure the same data is used across a SP group
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sp_size = self.device_mesh['sp'].size()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            group = self.device_mesh['sp'].get_group()

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prev_device = data.batch.device
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.batch = data.batch.cuda(device=torch.cuda.current_device())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.batch = allgather_dict_tensors(data.batch.contiguous(), size=sp_size, group=group, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.batch = data.batch.to(prev_device)
            # all gather non_tensor_batch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_non_tensor_batch = [None for _ in range(sp_size)]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.non_tensor_batch = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def postprocess_data(self, data: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Split the data to follow FSDP partition
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sp_size = self.device_mesh['sp'].size()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sp_rank = self.device_mesh['sp'].get_local_rank()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = data.chunk(chunks=sp_size)[sp_rank]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data