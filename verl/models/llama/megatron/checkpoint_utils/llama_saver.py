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
import megatron
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.utils import print_rank_0, unwrap_model
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.model import Float16Module
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.model import DistributedDataParallel as LocalDDP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.nn.parallel import DistributedDataParallel as torchDDP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron import get_args


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _megatron_calc_global_rank(tp_rank: int = 0, dp_rank: int = 0, pp_rank: int = 0):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """given TP,DP,PP rank to get the global rank."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = get_args()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tp_size = mpu.get_tensor_model_parallel_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_size = mpu.get_data_parallel_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert (tp_size * dp_size * pp_size == torch.distributed.get_world_size()
           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
           ), f"{tp_size} x {dp_size} x {pp_size} != {torch.distributed.get_world_size()}"
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if args.switch_dp_and_pp_grouping:
        # TP-PP-DP grouping
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (dp_rank * pp_size + pp_rank) * tp_size + tp_rank
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # TP-DP-PP grouping
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (pp_rank * dp_size + dp_rank) * tp_size + tp_rank


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _megatron_calc_layer_map(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Calculate the mapping of global layer_idx to local layer_idx
    Returns:
        layer_map (Dict: int -> tuple(int, int, int)):
            mapping from the global layer index to
            a tuple of (pp_rank, virtual_pp_rank, layer_idx inside model)
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import megatron
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.core import mpu

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = megatron.get_args()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    layer_map = dict()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for pp_rank_idx in range(pp_size):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for virtual_pp_rank_idx in range(virtual_pp_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_offset = (virtual_pp_rank_idx * (config.num_hidden_layers // virtual_pp_size) +
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            pp_rank_idx * num_layers_per_model)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for layer_idx in range(num_layers_per_model):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                layer_map[layer_offset + layer_idx] = (
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    pp_rank_idx,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    virtual_pp_rank_idx,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    layer_idx,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return layer_map


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def merge_megatron_ckpt_llama(wrapped_models, config, is_value_model=False, dtype='bf16'):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Merge sharded parameters of a Megatron module into a merged checkpoint.

    Args:
        wrapped_modelss (list of megatron.model.DistributedDataParallel):
            The local DDP wrapped megatron modules.
        dtype (str or None):
            The data type of state_dict. if None, the data type of the original parameters
            is used.
        gpt_model_key: key to access model
    Returns:
        state_dict (dict):
            The merged state_dict in rank 0, and an empty dictionary in other ranks.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    start_time = time.time()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = megatron.get_args()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_gpt_model(model):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return model

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_rank = mpu.get_data_parallel_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mp_group = mpu.get_model_parallel_group()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if dist.get_rank() == 0:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert mp_group.rank() == 0, f"mp_rank:[{mp_group.rank}] != 0 on rank #0"
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert pp_rank == 0, f"pp_rank:[{pp_rank}] != 0 on rank #0"
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert dp_rank == 0, f"dp_rank:[{dp_rank}] != 0 on rank #0"

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not isinstance(wrapped_models, (list, tuple)):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        wrapped_models = list(wrapped_models)

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert len(wrapped_models) == virtual_pp_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_layers_per_model = config.num_hidden_layers // pp_size // virtual_pp_size
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert num_layers_per_model * pp_size * virtual_pp_size == config.num_hidden_layers

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    models = [None] * len(wrapped_models)

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i, wrapped_model in enumerate(wrapped_models):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        models[i] = unwrap_model(wrapped_model, (torchDDP, LocalDDP, Float16Module))
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(models[i].model.layers
                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                  ) == num_layers_per_model, 'len model layers {} not equal to num_layers_per_model {}'.format(
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      len(models[i].model.layers), num_layers_per_model)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    state_dict = dict()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_cpu_tensor(tensor: torch.Tensor):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor is None:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor.device == torch.device("cpu"):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return tensor.detach().clone()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return tensor.detach().cpu()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tensor(tensor, name, src_pp_rank) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """broadcast tensor across mp_group"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal state_dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal mp_group
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == src_rank:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if tensor is None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight = None
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_shape = None
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight = tensor
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_shape = weight.shape
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight = None
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [tensor_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_shape = obj_list[0]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tensor:[{name}] not exist, skip collect")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if weight is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight = torch.empty(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=args.params_dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=torch.cuda.current_device(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                requires_grad=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast(weight, src=src_rank, group=mp_group)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[name] = _get_cpu_tensor(weight)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor(tensor, name, src_pp_rank, concat_dim=0, mutate_func=None) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """broadcast tensor in tp shards across mp_group"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal state_dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal mp_group
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == src_rank:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = tensor.shape
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{name}] not exist, skip collecting")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = torch.empty(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=args.params_dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device=torch.cuda.current_device(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            requires_grad=False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_tensors = [None] * tp_size

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            full_tensor = torch.concat(chunk_tensors, dim=concat_dim)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if mutate_func is not None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                full_tensor = mutate_func(full_tensor)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[name] = full_tensor

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name, src_pp_rank) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """broadcast tensor in tp shards across mp_group"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal state_dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal mp_group
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == src_rank:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = tensor.shape
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not exist, skip collecting")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = torch.empty(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=args.params_dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device=torch.cuda.current_device(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            requires_grad=False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_tensors = [None] * tp_size

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            full_tensor = torch.concat(chunk_tensors, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            intermediate_size_tp = config.intermediate_size // tp_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gate_weight_list = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            up_weight_list = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(tp_size):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gate_weight_list.append(gate_weight_tp)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                up_weight_list.append(up_weight_tp)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[gate_name] = torch.cat(gate_weight_list, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[up_name] = torch.cat(up_weight_list, dim=0)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name, src_pp_rank):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """broadcast tensor in tp shards across mp_group"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal state_dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal mp_group
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_rank = mpu.get_tensor_model_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=src_pp_rank)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == src_rank:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = tensor.shape
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=src_rank, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{q_name}] not exist, skip collecting")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = torch.empty(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=args.params_dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device=torch.cuda.current_device(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            requires_grad=False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_tensors = [None] * tp_size

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cur_src_rank = _megatron_calc_global_rank(tp_rank=i, dp_rank=0, pp_rank=src_pp_rank)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = tensor if torch.distributed.get_rank() == cur_src_rank else buffer_tensor
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=cur_src_rank, group=mp_group)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_tensors[i] = _get_cpu_tensor(sync_tensor)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            full_tensor = torch.concat(chunk_tensors, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            q_weight_list = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            k_weight_list = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            v_weight_list = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_size_per_head = config.hidden_size // config.num_attention_heads

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if config.num_key_value_heads >= tp_size:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                q_size_tp = config.hidden_size // tp_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                total_size = q_size_tp + 2 * kv_size_tp
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(tp_size):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    qkv_part = full_tensor[i * total_size:(i + 1) * total_size]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    q_part = qkv_part[:q_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    k_part = qkv_part[q_size_tp:q_size_tp + kv_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    v_part = qkv_part[q_size_tp + kv_size_tp:total_size]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    q_weight_list.append(q_part)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    k_weight_list.append(k_part)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    v_weight_list.append(v_part)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                q_size_tp = config.hidden_size // tp_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                kv_size_tp = hidden_size_per_head
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                total_size = q_size_tp + 2 * kv_size_tp
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(tp_size):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    qkv_part = full_tensor[i * total_size:(i + 1) * total_size]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    q_part = qkv_part[:q_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    k_part = qkv_part[q_size_tp:q_size_tp + kv_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    v_part = qkv_part[q_size_tp + kv_size_tp:total_size]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    q_weight_list.append(q_part)
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if i * config.num_key_value_heads % tp_size == 0:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        k_weight_list.append(k_part)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        v_weight_list.append(v_part)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[q_name] = torch.cat(q_weight_list, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[k_name] = torch.cat(k_weight_list, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict[v_name] = torch.cat(v_weight_list, dim=0)

    # empty cache before collecting weights
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()
    # Embeddings
    # -------------------
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if dp_rank == 0:
        # Embeddings
        # -------------------
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_rank_0("collecting embeddings...")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpt_model_module = _get_gpt_model(models[0])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _broadcast_tp_shard_tensor(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpt_model_module.model.embed_tokens.weight if pp_rank == 0 else None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "model.embed_tokens.weight",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            src_pp_rank=0,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # Transformer layers
        # -------------------
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        layer_map = _megatron_calc_layer_map(config)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for layer in range(config.num_hidden_layers):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"collecting layer #{layer}...")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_name = f"model.layers.{layer}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            src_pp_rank, src_virtual_pp_rank, src_layer_idx = layer_map[layer]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpt_model_module = _get_gpt_model(models[src_virtual_pp_rank])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_layer = gpt_model_module.model.layers[src_layer_idx]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.input_layernorm.weight,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.input_layernorm.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                src_pp_rank=src_pp_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor_qkv(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.self_attn.qkv_proj.weight,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.q_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.k_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.v_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                src_pp_rank=src_pp_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.self_attn.o_proj.weight,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.o_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                concat_dim=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                src_pp_rank=src_pp_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.post_attention_layernorm.weight,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.post_attention_layernorm.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                src_pp_rank=src_pp_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor_gate_up(sync_layer.mlp.gate_up_proj.weight,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               f"{layer_name}.mlp.gate_proj.weight",
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               f"{layer_name}.mlp.up_proj.weight",
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               src_pp_rank=src_pp_rank)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.mlp.down_proj.weight,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.mlp.down_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                concat_dim=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                src_pp_rank=src_pp_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # Final Layernorm
        # -------------------
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_rank_0("collecting final layernorm...")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpt_model_module = _get_gpt_model(models[-1])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _broadcast_tensor(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            getattr(gpt_model_module.model.norm, "weight", None),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "model.norm.weight",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            src_pp_rank=pp_size - 1,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_rank_0("collecting lm_head...")

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if is_value_model:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tensor(getattr(gpt_model_module.lm_head, "weight", None) if pp_rank == pp_size - 1 else None,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              "reward_head.weight",
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              src_pp_rank=pp_size - 1)

        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                getattr(gpt_model_module.lm_head, "weight", None) if pp_rank == pp_size - 1 else None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "lm_head.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                src_pp_rank=pp_size - 1,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dist.barrier()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.get_rank() == 0:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if dtype == "fp16":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype = torch.float16
        # 中文注释：下一行继续判断其他条件分支。
        elif dtype == "bf16":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype = torch.bfloat16
        # 中文注释：下一行继续判断其他条件分支。
        elif dtype is None or dtype == "fp32":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype = torch.float32
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Unknown/unsupported dtype to save: {dtype}"')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            exit(1)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for k, v in state_dict.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if dtype != v.dtype:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                state_dict[k] = v.to(dtype)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print_rank_0(f"merge megatron ckpt done, time elapsed {time.time() - start_time}s")
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return state_dict
