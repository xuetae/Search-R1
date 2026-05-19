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
Utilities for DeepSpeed Ulysses Sequence Parallelism.
DeepSpeed Ulysses Paper: https://arxiv.org/abs/2309.14509
Inspired from: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Any, Optional, List, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import Tensor
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed import ProcessGroup

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Set ulysses sequence parallel process group.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Get ulysses sequence parallel process group.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Get ulysses sequence parallel world size.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dist.get_world_size(group) if group else 1


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Get ulysses sequence parallel rank.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dist.get_rank(group) if group else 0


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gather_seq_scatter_heads(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    x: Tensor,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seq_dim: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    head_dim: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    unpadded_dim_size: int = 0,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group: ProcessGroup = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not group:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return x
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padding_size = x.size(seq_dim) - unpadded_dim_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        x = _unpad_tensor(x, seq_dim, padding_size)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not group:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return x
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dim_size = x.size(seq_dim)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if dim_size % sp_world != 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padding_size = sp_world - (dim_size % sp_world)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        x = _pad_tensor(x, seq_dim, padding_size)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    shape = list(x.shape)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    shape[dim] = padding_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.cat([x, pad], dim=dim)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    slc = [slice(None)] * len(x.shape)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    slc[dim] = slice(0, -padding_size)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x[slc]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_world_size = dist.get_world_size(group)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_rank = get_ulysses_sequence_parallel_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dim_size = x.size(dim)
    # pad before slice
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if padding and dim_size % sp_world_size:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padding_size = sp_world_size - (dim_size % sp_world_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parts = x.size(dim) // sp_world_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    slc = [slice(None)] * len(x.shape)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x[slc].contiguous()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def all_to_all_tensor(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_input: Tensor,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    scatter_dim: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    gather_dim: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group: Optional[dist.ProcessGroup] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    async_op: bool = False,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seq_world_size = dist.get_world_size(group)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if async_op:

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def wait():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            comm.wait()
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return torch.cat(output_list, dim=gather_dim).contiguous()

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return wait
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.cat(output_list, dim=gather_dim).contiguous()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def all_gather_tensor(local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_world_size = dist.get_world_size(group=group)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_shape = list(local_tensor.shape)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_shape[0] = output_shape[0] * sp_world_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义类，用于组织相关状态与行为。
class SeqAllToAll(torch.autograd.Function):

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx: Any,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group: dist.ProcessGroup,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_input: Tensor,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scatter_dim: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gather_dim: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        async_op: bool = False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.group = group
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.scatter_dim = scatter_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.gather_dim = gather_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.async_op = async_op
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ctx.async_op:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_t = grad_output[0]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Gather(torch.autograd.Function):

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(ctx: Any,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                group: dist.ProcessGroup,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                local_tensor: Tensor,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gather_dim: int,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                grad_scaler: bool = True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                async_op=False) -> Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.group = group
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.gather_dim = gather_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.grad_scaler = grad_scaler
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.async_op = async_op

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sp_world_size = dist.get_world_size(group=group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.sp_world_size = sp_world_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sp_rank = dist.get_rank(group=group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.sp_rank = sp_rank

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_shape = list(local_tensor.size())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        split_size = local_shape[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        part_size = local_shape[gather_dim]  # store original size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.part_size = part_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = all_gather_tensor(local_tensor, group, async_op)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ctx.grad_scaler:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_output = grad_output * ctx.sp_world_size
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (None, grad_output.split(ctx.part_size,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        dim=ctx.gather_dim)[ctx.sp_rank].contiguous(), None, None, None, None)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gather_outpus_and_unpad(x: Tensor,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            gather_dim: int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            unpad_dim: int = None,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            padding_size: int = 0,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            grad_scaler: bool = True,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            group: Optional[dist.ProcessGroup] = None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group = get_ulysses_sequence_parallel_group() if group is None else group
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sp_size = get_ulysses_sequence_parallel_world_size()
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if group == None:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return x
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if unpad_dim is not None:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(padding_size, int), 'padding size is not given or is not an integer'
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if padding_size == 0:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return x
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        x = _unpad_tensor(x, unpad_dim, padding_size)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def ulysses_pad_and_slice_inputs(input_ids_rmpad: torch.Tensor,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 position_ids_rmpad: Optional[torch.Tensor] = None,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 sp_size: int = 1):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded,
    but only input_ids will be sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size 
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if position_ids_rmpad is not None:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert position_ids_rmpad.size(0) == 1
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert input_ids_rmpad.size(1) == position_ids_rmpad.size(1)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if sp_size <= 1:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return input_ids_rmpad, position_ids_rmpad, 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _, total_seq_len = input_ids_rmpad.shape
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pad_size > 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if position_ids_rmpad is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    # we don't need to slice position ids
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return input_ids_rmpad, position_ids_rmpad, pad_size
