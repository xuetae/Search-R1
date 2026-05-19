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
from enum import Enum
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from functools import wraps
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, List, Tuple
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from types import FunctionType
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.protocol import DataProtoFuture

# here we add a magic number of avoid user-defined function already have this attribute
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
MAGIC_ATTR = 'attrs_3141562937'


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Dispatch(Enum):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    RANK_ZERO = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ONE_TO_ALL = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ALL_TO_ALL = 2
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON_COMPUTE = 3
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON_PP_AS_DP = 4
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON_PP_ONLY = 5
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON_COMPUTE_PROTO = 6
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON_PP_AS_DP_PROTO = 7
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DP_COMPUTE = 8
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DP_COMPUTE_PROTO = 9
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DP_COMPUTE_PROTO_WITH_FUNC = 10
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DP_COMPUTE_METRIC = 11


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Execute(Enum):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ALL = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    RANK_ZERO = 1


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.protocol import DataProto, DataProtoFuture
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_args = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arg in args:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(arg, (DataProto, DataProtoFuture))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        splitted_args.append(arg.chunk(chunks=chunks))

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_kwargs = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in kwargs.items():
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(val, (DataProto, DataProtoFuture))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return splitted_args, splitted_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_one_to_all(worker_group, *args, **kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    args = tuple([arg] * worker_group.world_size for arg in args)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return args, kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_all_to_all(worker_group, *args, **kwargs):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return args, kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_all_to_all(worker_group, output):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_megatron_compute(worker_group, *args, **kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    User passes in dp data. The data is dispatched to all tp/pp ranks with the same dp
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      MegatronWorkerGroup), f'worker_group must be MegatronWorkerGroup, Got {type(worker_group)}'

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_args = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arg in args:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(arg, (Tuple, List)) and len(arg) == worker_group.dp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformed_args = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(worker_group.world_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformed_args.append(arg[local_dp_rank])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_args.append(transformed_args)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_args = tuple(all_args)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_kwargs = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for k, v in kwargs.items():
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(v, (Tuple, List)) and len(v) == worker_group.dp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformed_v = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(worker_group.world_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformed_v.append(v[local_dp_rank])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_kwargs[k] = transformed_v
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return all_args, all_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_megatron_compute(worker_group, output):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Only collect the data from the tp=0 and pp=last and every dp ranks
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_in_dp = []
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = worker_group.get_megatron_global_info().pp_size
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for global_rank in range(worker_group.world_size):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output_in_dp.append(output[global_rank])
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output_in_dp


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_megatron_compute_data_proto(worker_group, *args, **kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _concat_data_proto_or_future(output: List):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.protocol import DataProto, DataProtoFuture
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import ray

    # make sure all the elements in output has the same type
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for o in output:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert type(o) == type(output[0])

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    o = output[0]

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(o, DataProto):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto.concat(output)
    # 中文注释：下一行继续判断其他条件分支。
    elif isinstance(o, ray.ObjectRef):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProtoFuture.concat(output)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_megatron_compute_data_proto(worker_group, output):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Each output must be a DataProto. We concat the dim=0 of output
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.protocol import DataProto
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import ray

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = collect_megatron_compute(worker_group, output)
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for o in output:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(o, (DataProto, ray.ObjectRef)), f"expecting {o} to be DataProto, but got {type(o)}"

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _concat_data_proto_or_future(output)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_megatron_pp_as_dp(worker_group, *args, **kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    treat pp as dp.
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = worker_group.pp_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_size = worker_group.dp_size

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_dp_size = pp_size * dp_size

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_args = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arg in args:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(arg, (List, Tuple)) and len(arg) == pp_dp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformed_args = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(worker_group.world_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # compute the rank in arg. Note that the order is dp then pp
            # Also note that the outputs within a pp group will be firstly allgathered, then only the output of pp0 will be collected.
            # For pp=2 dp=4, a batch of data "ABCDEFGH" should be dispatched and collected in below order:
            #    dispatch:       pp_allgther:        collect:
            #   dp 0 1 2 3      dp  0  1  2  3
            # pp +---------+  pp +-------------+
            #  0 | A C E G |   0 | AB CD EF GH |     ABCDEFGH
            #  1 | B D F H |   1 | AB CD EF GH |
            #    +---------+     +-------------+
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            arg_rank = local_dp_rank * worker_group.pp_size + local_pp_rank

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformed_args.append(arg[arg_rank])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_args.append(transformed_args)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_args = tuple(all_args)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_kwargs = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for k, v in kwargs.items():
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(v, (List, Tuple)) and len(v) == pp_dp_size, f'expect len(v)=={pp_dp_size}, got {len(v)}'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformed_v = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(worker_group.world_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # compute the rank in arg. Note that the order is dp then pp
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            arg_rank = local_dp_rank * worker_group.pp_size + local_pp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformed_v.append(v[arg_rank])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_kwargs[k] = transformed_v
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return all_args, all_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_megatron_pp_as_dp(worker_group, output):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    treat pp as dp. Only collect data on tp=0
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_in_dp = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for global_rank in range(worker_group.world_size):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output_in_dp.append(output[global_rank])
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output_in_dp


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_megatron_pp_only(worker_group, output):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Only collect output of megatron pp. This is useful when examine weight names as they are identical in tp/dp
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_in_pp = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for global_rank in range(worker_group.world_size):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if local_rank_info.tp_rank == 0 and local_rank_info.dp_rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output_in_pp.append(output[global_rank])
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output_in_pp


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_megatron_pp_as_dp_data_proto(worker_group, *args, **kwargs):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_dp_size = worker_group.dp_size * worker_group.pp_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(pp_dp_size, *args, **kwargs)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dispatch_megatron_pp_as_dp(worker_group, *splitted_args, **splitted_kwargs)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_megatron_pp_as_dp_data_proto(worker_group, output):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.protocol import DataProto
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, MegatronWorkerGroup)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = collect_megatron_pp_as_dp(worker_group, output)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _concat_data_proto_or_future(output)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_dp_compute(worker_group, *args, **kwargs):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.worker_group import WorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, WorkerGroup)
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arg in args:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(arg, (Tuple, List)) and len(arg) == worker_group.world_size
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for k, v in kwargs.items():
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(v, (Tuple, List)) and len(v) == worker_group.world_size
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return args, kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_dp_compute(worker_group, output):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.worker_group import WorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, WorkerGroup)
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert len(output) == worker_group.world_size
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.worker_group import WorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, WorkerGroup)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args, **kwargs)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return splitted_args, splitted_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.single_controller.base.worker_group import WorkerGroup
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(worker_group, WorkerGroup)
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert type(args[0]) == FunctionType  # NOTE: The first one args is a function!

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return splitted_args_with_func, splitted_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collect_dp_compute_data_proto(worker_group, output):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.protocol import DataProto
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import ray

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for o in output:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(o, (DataProto, ray.ObjectRef)), f"expecting {o} to be DataProto, but got {type(o)}"

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = collect_dp_compute(worker_group, output)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _concat_data_proto_or_future(output)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_predefined_dispatch_fn(dispatch_mode):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    predefined_dispatch_mode_fn = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.ONE_TO_ALL: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_one_to_all,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_all_to_all,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.ALL_TO_ALL: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_all_to_all,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_all_to_all,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.MEGATRON_COMPUTE: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_megatron_compute,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_megatron_compute,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.MEGATRON_PP_AS_DP: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_megatron_pp_as_dp,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_megatron_pp_as_dp,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.MEGATRON_PP_ONLY: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_one_to_all,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_megatron_pp_only
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.MEGATRON_COMPUTE_PROTO: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_megatron_compute_data_proto,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_megatron_compute_data_proto
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.MEGATRON_PP_AS_DP_PROTO: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_megatron_pp_as_dp_data_proto,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_megatron_pp_as_dp_data_proto
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.DP_COMPUTE: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_dp_compute,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_dp_compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.DP_COMPUTE_PROTO: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_dp_compute_data_proto,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_dp_compute_data_proto
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_dp_compute_data_proto_with_func,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_dp_compute_data_proto
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Dispatch.DP_COMPUTE_METRIC: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'dispatch_fn': dispatch_dp_compute_data_proto,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'collect_fn': collect_dp_compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return predefined_dispatch_mode_fn[dispatch_mode]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_predefined_execute_fn(execute_mode):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    predefined_execute_mode_fn = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Execute.ALL: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'execute_fn_name': 'execute_all'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        Execute.RANK_ZERO: {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'execute_fn_name': 'execute_rank_zero'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return predefined_execute_mode_fn[execute_mode]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _check_dispatch_mode(dispatch_mode):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(dispatch_mode,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      (Dispatch, Dict)), f'dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}'
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(dispatch_mode, Dict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        necessary_keys = ['dispatch_fn', 'collect_fn']
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in necessary_keys:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert key in dispatch_mode, f'key {key} should be in dispatch_mode if it is a dictionary'


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _check_execute_mode(execute_mode):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(execute_mode, Execute), f'execute_mode must be a Execute. Got {execute_mode}'


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _materialize_futures(*args, **kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    new_args = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arg in args:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(arg, DataProtoFuture):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            arg = arg.get()
        # add more type to materialize
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_args.append(arg)
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for k, v in kwargs.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(v, DataProtoFuture):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kwargs[k] = v.get()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    new_args = tuple(new_args)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return new_args, kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _check_execute_mode(execute_mode=execute_mode)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def decorator(func):

        # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
        @wraps(func)
        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def inner(*args, **kwargs):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if materialize_futures:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                args, kwargs = _materialize_futures(*args, **kwargs)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return func(*args, **kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attrs = {'dispatch_mode': dispatch_mode, 'execute_mode': execute_mode, 'blocking': blocking}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        setattr(inner, MAGIC_ATTR, attrs)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return inner

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return decorator
