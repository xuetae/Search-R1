# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""Model and data parallel groups."""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import vllm.model_executor.parallel_utils.parallel_state as ps
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
This version is strongly tied with Megatron to implement HybridEngine and weight sharing between vllm and Megatron.
- We assume the Megatron tp+dp+pp world is already established before calling this function.

"""

# Tensor model parallel group that the current rank belongs to.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_TENSOR_MODEL_PARALLEL_GROUP = None

# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_MICRO_DATA_PARALLEL_GROUP = None


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def initialize_model_parallel_from_megatron(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_model_parallel_size=None  # we set None for backward compatibility to set infer_tp = train_tp
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.core import parallel_state as mpu
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.distributed import new_group
    # Get world size and rank. Ensure some consistencies.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert torch.distributed.is_initialized()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tensor_model_parallel_size is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(tensor_model_parallel_size, int)

    # Build the tensor model-parallel groups.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert ps._TENSOR_MODEL_PARALLEL_GROUP is None, ("tensor model parallel group is already initialized")

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert tensor_model_parallel_size <= mpu.get_tensor_model_parallel_world_size(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ), 'Not implemented for infer_tp > train_tp'

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _TENSOR_MODEL_PARALLEL_GROUP
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _MICRO_DATA_PARALLEL_GROUP

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert mpu.get_tensor_model_parallel_world_size() % tensor_model_parallel_size == 0

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    micro_dp_size = mpu.get_tensor_model_parallel_world_size() // tensor_model_parallel_size

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    world_size: int = torch.distributed.get_world_size()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_micro_dp_groups = world_size // micro_dp_size

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank = torch.distributed.get_rank()

    # Build the micro dp groups.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(num_micro_dp_groups):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group = new_group(rank=rank, ranks=ranks, group_type='micro_dp')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if rank in ranks:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _MICRO_DATA_PARALLEL_GROUP = group

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tensor_model_parallel_size == mpu.get_tensor_model_parallel_world_size():
        # using the same tp group as Megatron
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ps._TENSOR_MODEL_PARALLEL_GROUP = mpu.get_tensor_model_parallel_group()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _TENSOR_MODEL_PARALLEL_GROUP = mpu.get_tensor_model_parallel_group()
        # no _MICRO_DATA_PARALLEL_GROUP
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # initialize a micro_dp group and a tp group
        # assume training tp=4, infer tp=2, then, weight is partitioned as
        # [1], [2], [3], [4] for training and [1,2], [1,2], [3,4], [3,4] for inference

        # Build the inference tp groups
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        train_tp = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_tensor_model_parallel_groups_per_train_tp = train_tp // tensor_model_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert _TENSOR_MODEL_PARALLEL_GROUP is None, ("tensor model parallel group is already initialized")
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            start = train_tp * i
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            end = train_tp * (i + 1)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(len(ranks)):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    ranks[i] += j
                # group = torch.distributed.new_group(ranks)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                group = new_group(rank=rank, ranks=ranks, group_type='infer_tp')
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if rank in ranks:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    _TENSOR_MODEL_PARALLEL_GROUP = group
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    ps._TENSOR_MODEL_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP
    # Build the pipeline model-parallel groups.
    # global _PIPELINE_MODEL_PARALLEL_GROUP
    # global _PIPELINE_GLOBAL_RANKS
    # assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")

    # ps._PIPELINE_MODEL_PARALLEL_GROUP = mpu.get_pipeline_model_parallel_group()
    # ps._PIPELINE_GLOBAL_RANKS = mpu.get_pipeline_model_parallel_ranks()


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Tensor model parallel utilities
"""


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_model_parallel_group():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Get the tensor model parallel group the caller rank belongs to."""
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, ("tensor model parallel group is not initialized")
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _TENSOR_MODEL_PARALLEL_GROUP


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_model_parallel_world_size():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Return world size for the tensor model parallel group."""
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_model_parallel_rank():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Return my rank for the tensor model parallel group."""
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_model_parallel_src_rank():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global_rank = torch.distributed.get_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_world_size = get_tensor_model_parallel_world_size()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return (global_rank // local_world_size) * local_world_size


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Micro Data parallel group
"""


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_micro_data_parallel_group():
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _MICRO_DATA_PARALLEL_GROUP is not None
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _MICRO_DATA_PARALLEL_GROUP


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_micro_data_parallel_world_size():
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.distributed.get_world_size(group=get_micro_data_parallel_group())


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_micro_data_parallel_rank():
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.distributed.get_rank(group=get_micro_data_parallel_group())
