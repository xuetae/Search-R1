# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""Model and data parallel groups."""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import vllm.distributed.parallel_state as ps
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.distributed.parallel_state import (
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    get_pp_group,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    get_world_group,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    init_distributed_environment,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    init_model_parallel_group,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
This version is strongly tied with Megatron to implement HybridEngine and weight sharing between vllm and Megatron.
- We assume the Megatron tp+dp+pp world is already established before calling this function.

"""

# Device mesh for using DTensor
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_DEVICE_MESH = None

# Tensor model parallel group that the current rank belongs to.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_TP = None
# Pipeline model parallel group that the current rank belongs to.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_PP = None


# This method is for initializing the ParallelGroup when using HybridEngine
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def initialize_parallel_state(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    distributed_init_method: str = "env://",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    backend: str = "nccl",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor_model_parallel_size: int = 1,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_tp_per_train_tp: int = 1,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pipeline_model_parallel_size: int = 1,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
):
    # torch.distributed.all_reduce does not free the input tensor until
    # the synchronization point. This causes the memory usage to grow
    # as the number of all_reduce calls increases. This env var disables
    # this behavior.
    # Related issue:
    # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank = int(os.getenv("RANK", "-1"))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Use the world_size set by TORCHRUN
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.get_world_size() > 1:
        # NOTE: build a sepearate inference group with infer tp & micro dp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        initialize_model_parallel_for_vllm(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_model_parallel_size=tensor_model_parallel_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_tensor_model_parallel_groups_per_train_tp=num_tp_per_train_tp,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def ensure_model_parallel_initialized(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor_model_parallel_size: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pipeline_model_parallel_size: int = 1,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    backend: Optional[str] = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # get the backend of _DEVICE_WORLD_GROUP
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not model_parallel_is_initialized():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "tensor parallel group already initialized, but of unexpected size: "
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f"{get_tensor_model_parallel_world_size()=} vs. "
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f"{tensor_model_parallel_size=}")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_world_size = get_pp_group().world_size
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert pp_world_size == pipeline_model_parallel_size, (
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "pipeline parallel group already initialized, but of unexpected size: "
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f"{pp_world_size=} vs. "
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f"{pipeline_model_parallel_size=}")


# TODO(sgm): deviate from the v0.5.4, not pp now
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def model_parallel_is_initialized():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Check if tensor and pipeline parallel groups are initialized."""
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return ps._TP is not None
    # and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def initialize_model_parallel_for_vllm(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor_model_parallel_size: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_tensor_model_parallel_groups_per_train_tp: int = 1,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pipeline_model_parallel_size: int = 1,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pass

    # Get world size and rank. Ensure some consistencies.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert torch.distributed.is_initialized()

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(tensor_model_parallel_size, int)

    # assert num_tensor_model_parallel_groups_per_train_tp == 1 and not different_tp_group
    # assert num_tensor_model_parallel_groups_per_train_tp > 1 and different_tp_group

    # Build the tensor model-parallel groups.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert ps._TP is None, "tensor model parallel group is already initialized"

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _TP

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    world_size: int = torch.distributed.get_world_size()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank = torch.distributed.get_rank()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    backend = torch.distributed.get_backend()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if num_tensor_model_parallel_groups_per_train_tp == 1:
        # if tensor_model_parallel_size == train_tensor_parallel_size:
        # using the same tp group as Megatron/vllm
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert _TP is None, "tensor model parallel group is already initialized"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group_ranks = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(num_tensor_model_parallel_groups):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            group_ranks.append(ranks)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _TP = init_model_parallel_group(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            group_ranks=group_ranks,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_rank=get_world_group().local_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            backend=backend,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_custom_allreduce=False,  # TODO: check why True is not work in Ray trainer
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_message_queue_broadcaster=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ps._TP = _TP
        # _MICRO_DATA_PARALLEL_GROUP is move to hybrid engine
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # initialize a micro_dp group and a tp group
        # assume training tp=4, infer tp=2, then, weight is partitioned as
        # [1], [2], [3], [4] for training and [1,2], [1,2], [3,4], [3,4] for inference

        # Build the inference tp groups
        # train_tp = train_tensor_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        train_tp = num_tensor_model_parallel_groups_per_train_tp * tensor_model_parallel_size
        # num_tensor_model_parallel_groups_per_train_tp = train_tp // tensor_model_parallel_size
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert _TP is None, "tensor model parallel group is already initialized"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group_ranks = []
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
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                group_ranks.append(ranks)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _TP = init_model_parallel_group(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            group_ranks=group_ranks,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_rank=get_world_group().local_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            backend=backend,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_custom_allreduce=False,  # TODO: check why True is not work in Ray trainer
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_message_queue_broadcaster=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ps._TP = _TP

    # Build the pipeline model-parallel groups.
    # global _PIPELINE_MODEL_PARALLEL_GROUP
    # global _PIPELINE_GLOBAL_RANKS
    # assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")

    # ps._PIPELINE_MODEL_PARALLEL_GROUP = mpu.get_pipeline_model_parallel_group()
    # ps._PIPELINE_GLOBAL_RANKS = mpu.get_pipeline_model_parallel_ranks()

    # TODO: init using device mesh (not support hybrid engine now)
    # Build the pipeline model-parallel groups.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _PP
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _PP is None, "pipeline model parallel group is already initialized"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group_ranks = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(num_pipeline_model_parallel_groups):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ps._PP = _PP  # for verl


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def initialize_model_parallel(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor_model_parallel_size: int = 1,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pipeline_model_parallel_size: int = 1,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    backend: Optional[str] = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    NOTE: This method is a hack from the open-sourced version without
    asertion of world_size = tp * pp

    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert torch.distributed.is_initialized()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    world_size: int = torch.distributed.get_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    backend = backend or torch.distributed.get_backend(ps.get_world_group().device_group)

    # NOTE(sgm) we don't assert world_size == tp * pp
    # DP is not managed by vllm but by the VeRL WorkerGroup
    # if (world_size !=
    #         tensor_model_parallel_size * pipeline_model_parallel_size):
    #     raise RuntimeError(
    #         f"world_size ({world_size}) is not equal to "
    #         f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
    #         f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank = torch.distributed.get_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _TP
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _TP is None, "tensor model parallel group is already initialized"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group_ranks = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(num_tensor_model_parallel_groups):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _TP = init_model_parallel_group(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group_ranks,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        get_world_group().local_rank,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        backend,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_custom_allreduce=False,  # TODO: check why True is not work in Ray trainer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_message_queue_broadcaster=True,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ps._TP = _TP

    # TODO: init using device mesh (not support hybrid engine now)
    # Build the pipeline model-parallel groups.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    global _PP
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _PP is None, "pipeline model parallel group is already initialized"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    group_ranks = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(num_pipeline_model_parallel_groups):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ps._PP = _PP  # for verl


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Device mesh utilities
"""


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_device_mesh():
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _DEVICE_MESH is not None, "device mesh is not initialized"
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _DEVICE_MESH


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Tensor model parallel utilities
"""


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_model_parallel_group():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Get the tensor model parallel group the caller rank belongs to."""
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _TP is not None, "tensor model parallel group is not initialized"
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _TP.device_group


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
