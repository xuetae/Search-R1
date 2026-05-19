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
the class for Worker
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import socket
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.decorator import register, Dispatch, Execute


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class DistRankInfo:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tp_rank: int
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_rank: int
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_rank: int


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class DistGlobalInfo:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tp_size: int
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_size: int
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size: int


# 中文注释：下一行定义类，用于组织相关状态与行为。
class WorkerHelper:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_node_ip(self):

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def get_node_ip_by_sdk():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if os.getenv("WG_BACKEND", None) == "ray":
                # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
                import ray
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return ray._private.services.get_node_ip_address()
            # 中文注释：下一行继续判断其他条件分支。
            elif os.getenv("WG_BACKEND", None) == "torch_rpc":
                # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
                from verl.single_controller.torchrpc.k8s_client import get_ip_addr
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return get_ip_addr()
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        host_ipv4 = os.getenv("MY_HOST_IP", None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        host_ip_by_env = host_ipv4 or host_ipv6
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        host_ip_by_sdk = get_node_ip_by_sdk()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        host_ip = host_ip_by_env or host_ip_by_sdk
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return host_ip

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_free_port(self):
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with socket.socket() as sock:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sock.bind(('', 0))
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return sock.getsockname()[1]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_availale_master_addr_port(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._get_node_ip(), str(self._get_free_port())

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_pid(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return


# 中文注释：下一行定义类，用于组织相关状态与行为。
class WorkerMeta:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    keys = [
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, store) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._store = store

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def to_dict(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {f"_{key.lower()}": self._store.get(f"_{key.lower()}", None) for key in WorkerMeta.keys}


# we assume that in each WorkerGroup, there is a Master Worker
# 中文注释：下一行定义类，用于组织相关状态与行为。
class Worker(WorkerHelper):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __new__(cls, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        instance = super().__new__(cls)

        # note that here we use int to distinguish
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        disable_worker_init = int(os.environ.get('DISABLE_WORKER_INIT', 0))
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if disable_worker_init:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return instance

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = os.environ.get("RANK", None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        # when decorator @ray.remote applies, __new__ will be called while we don't want to apply _configure_before_init
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return instance

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _configure_before_init(self, register_center_name: str, rank: int):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            master_addr, master_port = self.get_availale_master_addr_port()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank_zero_info = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "MASTER_ADDR": master_addr,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "MASTER_PORT": master_port,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if os.getenv("WG_BACKEND", None) == "ray":
                # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.register_center = create_worker_group_register_center(name=register_center_name,
                                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                           info=rank_zero_info)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.environ.update(rank_zero_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, cuda_visible_devices=None) -> None:
        # construct a meta from envrionment variable. Note that the import must be inside the class because it is executed remotely
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import os
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = int(os.environ['WORLD_SIZE'])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = int(os.environ['RANK'])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._rank = rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._world_size = world_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        master_addr = os.environ["MASTER_ADDR"]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        master_port = os.environ["MASTER_PORT"]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        store = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            '_world_size': world_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            '_rank': rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            '_local_world_size': local_world_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            '_local_rank': local_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            '_master_addr': master_addr,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            '_master_port': master_port
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if cuda_visible_devices is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            store['_cuda_visible_devices'] = cuda_visible_devices

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta = WorkerMeta(store=store)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._configure_with_meta(meta=meta)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _configure_with_meta(self, meta: WorkerMeta):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        This function should only be called inside by WorkerGroup
        """
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(meta, WorkerMeta)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.__dict__.update(meta.to_dict())  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in WorkerMeta.keys:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            val = self.__dict__.get(f"_{key.lower()}", None)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if val is not None:
                # print(f"set {key} to {val}")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                os.environ[key] = str(val)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "]", "") if self._master_addr else ""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_master_addr_port(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._master_addr, self._master_port

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_cuda_visible_devices(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import os
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return cuda_visible_devices

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def world_size(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._world_size

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def rank(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._rank

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_with_func_generator(self, func, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ret_proto = func(self, *args, **kwargs)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ret_proto

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_func_rank_zero(self, func, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        result = func(*args, **kwargs)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return result