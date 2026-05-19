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
This file contains utilities to manipulate torch memory buffers
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, List

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MemoryBuffer:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    A memory buffer is a contiguous torch tensor that may combine multiple tensors sharing with the underlying
    memory. It must have a unique type to support this behavior.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, numel: int, numel_padded: int, dtype: torch.dtype):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.numel = numel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.numel_padded = numel_padded
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dtype = dtype
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.data = torch.zeros(self.numel_padded, dtype=self.dtype, device='cuda', requires_grad=False)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def zero(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Reset the buffer to zero."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.data.zero_()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get(self, shape, start_index):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        end_index = start_index + shape.numel()
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = self.data[start_index:end_index]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = buffer_tensor.view(shape)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return buffer_tensor


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def calc_padded_numel(shape: torch.Size, dtype: torch.dtype):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """for cuda memory alignment, make sure alignment by 128-bits"""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    align_numel = 128 // torch.finfo(dtype).bits
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    numel = shape.numel()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return (numel + align_numel - 1) // align_numel * align_numel


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_weight_buffer_meta_from_module(module: nn.Module) -> Dict[str, Dict]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Return a dictionary containing name to a shape and dtype.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_buffer_meta = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, param in sorted(module.named_parameters()):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return weight_buffer_meta


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def build_memory_buffer(weight_buffer_meta: Dict[str, Dict]) -> Dict[torch.dtype, MemoryBuffer]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Build the memory buffer given weight_buffer_meta

    Args:
        weight_buffer_meta: contains mapping from name to a dictionary containing shape and dtype of the tensors

    Returns: a large memory buffer for each dtype that can hold all the tensors

    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    memory_buffers = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_numel_map = {}  # map from dtype to the total numel
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, meta_info in sorted(weight_buffer_meta.items()):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shape = meta_info['shape']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dtype = meta_info['dtype']

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(shape, torch.Size)
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(dtype, torch.dtype)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if dtype not in total_numel_map:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_numel_map[dtype] = 0

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_numel_map[dtype] += calc_padded_numel(shape, dtype)

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for dtype, total_numel in total_numel_map.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memory_buffers[dtype] = MemoryBuffer(total_numel, total_numel, dtype)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return memory_buffers


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def build_memory_reference_from_module(module: torch.nn.Module,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       memory_buffers: Dict[torch.dtype, MemoryBuffer],
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       maintain_weight=True):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    start_index = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for dtype in memory_buffers.keys():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        start_index[dtype] = 0
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, param in sorted(module.named_parameters()):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        memory_buffer = memory_buffers[param.dtype]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer = memory_buffer.get(shape=param.shape, start_index=start_index[param.dtype])
        # need to increment start_index
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        start_index[param.dtype] += calc_padded_numel(param.shape, dtype)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if maintain_weight:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer.copy_(param.data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param.data = buffer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def build_memory_reference(weight_buffer_meta: Dict[str, Dict], memory_buffers: Dict[torch.dtype, MemoryBuffer]):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Build the memory references. The memory buffers are built using the build_memory_buffer API.
    This API will allocate a weight buffer pointer to the memory buffer according to the weight_buffer_meta.

    Args:
        weight_buffer_meta:
        memory_buffers:

    Returns:

    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    start_idx = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_buffers = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for dtype in memory_buffers.keys():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        start_idx[dtype] = 0

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, meta_info in sorted(weight_buffer_meta.items()):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shape = meta_info['shape']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dtype = meta_info['dtype']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer = memory_buffers[dtype].get(shape, start_index=start_idx[dtype])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        start_idx[dtype] += calc_padded_numel(shape, dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_buffers[name] = buffer

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return weight_buffers


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MemoryBufferModuleWrapper:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Note that we do not design MemoryBufferModuleWrapper as an nn.Module due to
    - It will change the checkpoint name
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, module: nn.Module):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module = module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.weight_buffer_meta = get_weight_buffer_meta_from_module(self.module)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.memory_buffers = build_memory_buffer(self.weight_buffer_meta)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        build_memory_reference_from_module(self.module, self.memory_buffers)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_memory_buffers(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.memory_buffers

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_weight_buffer_meta(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.weight_buffer_meta


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronMemoryBufferForRollout(object):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    We assume that
    - inference engine has tp + dp
    - actor has tp + pp + dp
    - the tp between inference engine and actor should be the same
    - memory_buffers: contains a list of memory_buffers, each is a dict from dtype to MemoryBuffer
    - weight_buffers: contains a list of weight_buffers, each is a dict from name to param
    - named_parameters: a dict from name to parameter that normalizes the names from pp and vpp. Note that
        the named_parameters may not be directly compatible with inference engine. User has to take care of
        this part such as the layout mismatches. (e.g. qkv transpose)
    - Note that weight_buffer, named_parameters and memory_buffers share the same underlying GPU memory.
    - When doing weight sync, the data is transfer via memory buffers
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, transform_memory_param_fn):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._memory_buffers = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._weight_buffers = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._named_parameters = {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.transform_memory_param_fn = transform_memory_param_fn

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def initialize_weight_buffer(self, weight_buffer_meta_pp: List[Dict[str, Dict]]):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Initialize the weight buffer. The weight buffer is obtained according to the actor. We will construct
        a large buffer for each dtype in the weight_buffer.

        Args:
            weight_buffer_meta: contains pp models, each pp models contains a dictionary of mapping from

        Returns: None

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.weight_buffer_meta_pp = weight_buffer_meta_pp

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for weight_buffer_meta in self.weight_buffer_meta_pp:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            memory_buffer = build_memory_buffer(weight_buffer_meta)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._memory_buffers.append(memory_buffer)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._weight_buffers.append(None)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def build_memory_reference(self):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, weight_buffer_meta in enumerate(self.weight_buffer_meta_pp):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._weight_buffers[i] = build_memory_reference(weight_buffer_meta, self._memory_buffers[i])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._named_parameters = self.transform_memory_param_fn(self._weight_buffers)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def named_parameters(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._named_parameters

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def weight_buffers(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._weight_buffers

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def memory_buffers(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._memory_buffers
