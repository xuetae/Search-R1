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
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Any, Callable, Optional
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist


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
def load_state_dict_to_megatron_llama(state_dict, wrapped_models, config, params_dtype, is_value_model=False):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Load merged state_dict to sharded Megatron module in training.
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import megatron
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.core import mpu
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.utils import print_rank_0, unwrap_model
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.core.transformer.module import Float16Module
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.core import DistributedDataParallel as LocalDDP
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from torch.nn.parallel import DistributedDataParallel as torchDDP

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    start_time = time.time()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_gpt_model(model):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return model

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def broadcast_params(module):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for param in module.parameters():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.broadcast(param.data,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        src=mpu.get_data_parallel_src_rank(),
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        group=mpu.get_data_parallel_group())

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_rank = mpu.get_data_parallel_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    virtual_pp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mp_group = mpu.get_model_parallel_group()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.get_rank() == 0:
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
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpt_model_module = _get_gpt_model(models[i])
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(gpt_model_module.model.layers) == num_layers_per_model

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tensor(tensor, name) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """broadcast tensor from rank0 across mp_group"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal state_dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nonlocal mp_group
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name in state_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                weight = state_dict[name]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_shape = weight.shape
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_shape = None
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            weight = None
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [tensor_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_shape = obj_list[0]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tensor:[{name}] not in state_dict, skip load")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor = torch.empty(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=params_dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=torch.cuda.current_device(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                requires_grad=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor.data.copy_(weight)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast(tensor, src=0, group=mp_group)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor_vocab(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
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

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name in state_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                full_weight = state_dict[name]

                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if mutate_func is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    full_weight = mutate_func(full_weight)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape = tensor_chunk[0].shape
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape = None
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=params_dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=torch.cuda.current_device(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                requires_grad=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert (tensor.shape == chunk_shape
                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                   ), f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_tensor.data.copy_(tensor_chunk[i])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (i == tp_rank) and (tensor is not None):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor.data.copy_(sync_tensor)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor(tensor, name, chunk_dim=0, mutate_func=None) -> torch.Tensor:
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

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if name in state_dict:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                full_weight = state_dict[name]
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if mutate_func is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    full_weight = mutate_func(full_weight)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_chunk = torch.chunk(full_weight, tp_size, dim=chunk_dim)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape = tensor_chunk[0].shape
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape = None
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=params_dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=torch.cuda.current_device(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                requires_grad=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert (tensor.shape == chunk_shape
                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                   ), f"rank #{torch.distributed.get_rank()} tensor {name} shape {tensor.shape} != {chunk_shape}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_tensor.data.copy_(tensor_chunk[i])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (i == tp_rank) and (tensor is not None):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor.data.copy_(sync_tensor)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor_gate_up(tensor, gate_name, up_name) -> torch.Tensor:
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

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gate_weight = state_dict[gate_name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            up_weight = state_dict[up_name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_gate_up_weight = torch.empty(config.intermediate_size * 2,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             config.hidden_size,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             dtype=params_dtype,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             device=torch.cuda.current_device())
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(tp_size):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                intermediate_size_tp = config.intermediate_size // tp_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gate_weight_tp = gate_weight[i * intermediate_size_tp:(i + 1) * intermediate_size_tp]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                up_weight_tp = up_weight[i * intermediate_size_tp:(i + 1) * intermediate_size_tp]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                new_gate_up_weight[intermediate_size_tp * 2 * i:intermediate_size_tp * 2 * (i + 1)].copy_(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    torch.cat([gate_weight_tp, up_weight_tp], dim=0))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_chunk = torch.chunk(new_gate_up_weight, tp_size, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = tensor_chunk[0].shape
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{gate_name, up_name}] not in state_dict, skip loading")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=params_dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=torch.cuda.current_device(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                requires_grad=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert (
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor.shape == chunk_shape
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ), f"rank #{torch.distributed.get_rank() == 0:} tensor {gate_name, up_name} shape {tensor.shape} != {chunk_shape}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_tensor.data.copy_(tensor_chunk[i])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (i == tp_rank) and (tensor is not None):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor.data.copy_(sync_tensor)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _broadcast_tp_shard_tensor_qkv(tensor, q_name, k_name, v_name) -> torch.Tensor:
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

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert (q_name in state_dict and k_name in state_dict and v_name in state_dict)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            full_weight_q = state_dict[q_name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            full_weight_k = state_dict[k_name]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            full_weight_v = state_dict[v_name]

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
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                new_weight_qkv = torch.empty(total_size * tp_size,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             config.hidden_size,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             dtype=params_dtype,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             device=torch.cuda.current_device())
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(tp_size):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    q_part = full_weight_q[i * q_size_tp:(i + 1) * q_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    k_part = full_weight_k[i * kv_size_tp:(i + 1) * kv_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    v_part = full_weight_v[i * kv_size_tp:(i + 1) * kv_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    new_weight_qkv[i * total_size:(i + 1) * total_size].copy_(torch.cat([q_part, k_part, v_part],
                                                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                                        dim=0))

            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                q_size_tp = config.hidden_size // tp_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                kv_size_tp = hidden_size_per_head
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                total_size = q_size_tp + 2 * kv_size_tp
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                new_weight_qkv = torch.empty(total_size * tp_size,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             config.hidden_size,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             dtype=params_dtype,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             device=torch.cuda.current_device())
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(tp_size):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    q_part = full_weight_q[i * q_size_tp:(i + 1) * q_size_tp]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    start_idx = i * config.num_key_value_heads // tp_size * hidden_size_per_head
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    end_idx = (i * config.num_key_value_heads // tp_size + 1) * hidden_size_per_head
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    k_part = full_weight_k[start_idx:end_idx]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    v_part = full_weight_v[start_idx:end_idx]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    new_weight_qkv[i * total_size:(i + 1) * total_size].copy_(torch.cat([q_part, k_part, v_part],
                                                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                                        dim=0))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_chunk = torch.chunk(new_weight_qkv, tp_size, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = tensor_chunk[0].shape
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chunk_shape = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        obj_list = [chunk_shape]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.broadcast_object_list(obj_list, src=0, group=mp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chunk_shape = obj_list[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if chunk_shape is None:
            # all or none ranks in the mp_group should reach here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"tp_shard tensor:[{name}] not in state_dict, skip loading")
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensor is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_shape,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dtype=params_dtype,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                device=torch.cuda.current_device(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                requires_grad=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert (tensor.shape == chunk_shape
                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                   ), f"rank #{torch.distributed.get_rank()} tensor {q_name} shape {tensor.shape} != {chunk_shape}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_tensor = torch.empty_like(tensor, device=torch.cuda.current_device(), requires_grad=False)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(tp_size):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_rank() == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_tensor.data.copy_(tensor_chunk[i])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(sync_tensor, src=0, group=mp_group)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (i == tp_rank) and (tensor is not None):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor.data.copy_(sync_tensor)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if dp_rank == 0:
        # Embeddings
        # -------------------
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_rank_0("loading embeddings...")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpt_model_module = _get_gpt_model(models[0])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        embed_tokens_weight = None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if pp_rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            embed_tokens_weight = gpt_model_module.model.embed_tokens.weight
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _broadcast_tp_shard_tensor_vocab(embed_tokens_weight, "model.embed_tokens.weight")

        # Transformer layers
        # -------------------
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        layer_map = _megatron_calc_layer_map(config)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for layer in range(config.num_hidden_layers):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_rank_0(f"loading layer #{layer}...")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_name = f"model.layers.{layer}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dst_pp_rank, dst_virtual_pp_rank, dst_layer_idx = layer_map[layer]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpt_model_module = _get_gpt_model(models[dst_virtual_pp_rank])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_layer = gpt_model_module.model.layers[dst_layer_idx]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.input_layernorm.weight if dst_pp_rank == pp_rank else None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.input_layernorm.weight",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor_qkv(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.self_attn.qkv_proj.weight if dst_pp_rank == pp_rank else None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.q_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.k_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.v_proj.weight",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.self_attn.o_proj.weight if dst_pp_rank == pp_rank else None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.self_attn.o_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_dim=1,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.post_attention_layernorm.weight if dst_pp_rank == pp_rank else None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.post_attention_layernorm.weight",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor_gate_up(sync_layer.mlp.gate_up_proj.weight if dst_pp_rank == pp_rank else None,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               f"{layer_name}.mlp.gate_proj.weight", f"{layer_name}.mlp.up_proj.weight")

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sync_layer.mlp.down_proj.weight if dst_pp_rank == pp_rank else None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"{layer_name}.mlp.down_proj.weight",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk_dim=1,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # Final Layernorm
        # -------------------
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_rank_0("loading final layernorm...")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpt_model_module = _get_gpt_model(models[-1])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _broadcast_tensor(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            getattr(gpt_model_module.model.norm, "weight", None),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "model.norm.weight",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_rank_0("loading lm_head...")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lm_head_weight = None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if pp_rank + 1 == pp_size:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lm_head_weight = gpt_model_module.lm_head.weight

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if is_value_model:
            # if torch.distributed.get_rank() == 0:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if 'lm_head.weight' in state_dict and state_dict['lm_head.weight'].shape[0] == 1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                _broadcast_tensor(lm_head_weight, "lm_head.weight")
            # 中文注释：下一行继续判断其他条件分支。
            elif 'reward_head.weight' in state_dict and state_dict['reward_head.weight'].shape[0] == 1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                _broadcast_tensor(lm_head_weight, "reward_head.weight")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print_rank_0('load lm_head from value_head weight')
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                _broadcast_tensor(None, "lm_head.weight")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print_rank_0('fail to match lm_head in value_model')
            # else:

            #     _broadcast_tensor(lm_head_weight, "lm_head.weight")

        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _broadcast_tp_shard_tensor(lm_head_weight, "lm_head.weight")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dist.barrier()
    # Broadcast weights inside data parallel groups
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for wrapped_model in wrapped_models:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        broadcast_params(wrapped_model)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print_rank_0(f"loading megatron ckpt done, time elapsed {time.time() - start_time}s")
