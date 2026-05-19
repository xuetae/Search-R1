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
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, List, Any, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from ray.util import list_named_actors
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from ray.util.placement_group import placement_group, PlacementGroup
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from ray.experimental.state.api import get_actor

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base import WorkerGroup, ResourcePool, ClassWithInitArgs, Worker

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['Worker']


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_random_string(length: int) -> str:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import random
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import string
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    letters_digits = string.ascii_letters + string.digits
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return ''.join(random.choice(letters_digits) for _ in range(length))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def func(*args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = execute_fn(method_name, *args, **kwargs)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if blocking:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = ray.get(output)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = collect_fn(self, output)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return func


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RayResourcePool(ResourcePool):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 process_on_nodes: List[int] = None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 use_gpu: bool = True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 name_prefix: str = "",
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 max_colocate_count: int = 5,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 detached=False) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(process_on_nodes, max_colocate_count)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_gpu = use_gpu
        # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.name_prefix = name_prefix
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pgs = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.detached = detached

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_placement_groups(self, strategy="STRICT_PACK", name=None):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.pgs is not None:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.pgs

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pg_name_prefix = name if name else \
            f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        # print(f"pg_name_prefix = {pg_name_prefix}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pg_scheme = [[{
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "CPU": self.max_collocate_count,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "GPU": 1
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        } if self.use_gpu else {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "CPU": self.max_collocate_count
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        } for _ in range(process_count)] for process_count in self._store]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lifetime = 'detached' if self.detached else None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pgs = [
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for idx, bundles in enumerate(pg_scheme)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ray.get([pg.ready() for pg in pgs])

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pgs = pgs
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return pgs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_pg_from_exist(resource_pools: Dict[str, RayResourcePool], src_role_names: List[str],
                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                          resource_pool: RayResourcePool) -> List:

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    src_pgs = [
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pg for role_name, resource_pool in resource_pools.items() for pg in resource_pool.get_placement_groups()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if role_name in src_role_names
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    unsorted_pgs: List[Tuple[int, PlacementGroup]] = []
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    searching_idx = 0
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for request_process, original_idx in sorted_process_on_nodes:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, \
            f"requesting {request_process} processes, bundle count cannot satisfy"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        searching_idx += 1

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return [pg for _, pg in sorted(unsorted_pgs)]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert rp1.use_gpu == rp2.use_gpu, 'Both RayResourcePool must either use_gpu or not'
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert rp1.max_collocate_count == rp2.max_collocate_count, 'Both RayResourcePool must has the same max_collocate_count'
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, 'Both RayResourcePool must has the same n_gpus_per_node'
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert rp1.detached == rp2.detached, 'Detached ResourcePool cannot be merged with non-detached ResourcePool'

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    new_store = rp1.store + rp2.store

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    merged = RayResourcePool(new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    merged.pgs = rp1.get_placement_groups() + rp2.get_placement_groups()

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return merged


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RayClassWithInitArgs(ClassWithInitArgs):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(cls, *args, **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._options = {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._additional_resource = {}

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def set_additional_resource(self, additional_resource):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._additional_resource = additional_resource

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_options(self, options: Dict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._options.update(options)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 placement_group,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 placement_group_bundle_idx,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 use_gpu: bool = True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 num_gpus=1,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 sharing_with=None) -> Any:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if sharing_with is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cuda_visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.cls.options(**options).remote(*self.args,
                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                      cuda_visible_devices=cuda_visible_devices,
                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                      **self.kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        options = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "scheduling_strategy":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                PlacementGroupSchedulingStrategy(placement_group=placement_group,
                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                 placement_group_bundle_index=placement_group_bundle_idx)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        options.update(self._options)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_gpu:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            options["num_gpus"] = num_gpus

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(self._additional_resource) > 1:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for k, v in self._additional_resource.items():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RayWorkerGroup(WorkerGroup):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 resource_pool: RayResourcePool = None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 ray_cls_with_init: RayClassWithInitArgs = None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 bin_pack: bool = True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 name_prefix: str = None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 detached=False,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 worker_names=None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 **kwargs) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(resource_pool=resource_pool, **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ray_cls_with_init = ray_cls_with_init
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if worker_names is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert self._is_init_with_detached_workers
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._worker_names = worker_names

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_init_with_detached_workers:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._init_with_detached_workers(worker_names=worker_names)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._init_with_resource_pool(resource_pool=resource_pool,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          ray_cls_with_init=ray_cls_with_init,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          bin_pack=bin_pack,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          detached=detached)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ray_cls_with_init is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        worker_state_dict = get_actor(worker._actor_id.hex())
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_with_detached_workers(self, worker_names):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        workers = [ray.get_actor(name=name) for name in worker_names]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._workers = workers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._world_size = len(worker_names)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_gpu = resource_pool.use_gpu

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        strategy = "PACK"
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if bin_pack:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            strategy = "STRICT_PACK"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pgs = resource_pool.get_placement_groups(strategy=strategy)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = resource_pool.world_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpus = 1 / resource_pool.max_collocate_count

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = -1
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for pg_idx, local_world_size in enumerate(resource_pool.store):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pg = pgs[pg_idx]
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert local_world_size <= pg.bundle_count, \
                f"when generating for {self.name_prefix}, for the "
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for local_rank in range(local_world_size):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rank += 1

                # we pass in environment variable at option so that Worker can use environment variable to set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                env_vars = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'WORLD_SIZE': str(world_size),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'RANK': str(rank),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'WG_PREFIX': self.name_prefix,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'WG_BACKEND': 'ray',
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'RAY_LOCAL_WORLD_SIZE': str(local_world_size),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'RAY_LOCAL_RANK': str(local_rank),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if rank != 0:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    env_vars['MASTER_ADDR'] = self._master_addr
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    env_vars['MASTER_PORT'] = self._master_port

                # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
                import re
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                cia_name = type(ray_cls_with_init.cls).__name__
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ray_cls_with_init.update_options({'runtime_env': {'env_vars': env_vars}, 'name': name})

                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if detached:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    ray_cls_with_init.update_options({'lifetime': 'detached'})

                # create a worker
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                worker = ray_cls_with_init(placement_group=pg,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           placement_group_bundle_idx=local_rank,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           use_gpu=use_gpu,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           num_gpus=num_gpus)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._workers.append(worker)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._worker_names.append(name)

                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if rank == 0:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    register_center_actor = None
                    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                    for _ in range(120):
                        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                        if f"{self.name_prefix}_register_center" not in list_named_actors():
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            time.sleep(1)
                        # 中文注释：下一行处理前面条件都不满足时的默认分支。
                        else:
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            register_center_actor = ray.get_actor(f"{self.name_prefix}_register_center")
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            break
                    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                    assert register_center_actor is not None, f"failed to get register_center_actor: {self.name_prefix}_register_center in {list_named_actors(all_namespaces=True)}"
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self._master_addr, self._master_port = rank_zero_info['MASTER_ADDR'], rank_zero_info['MASTER_PORT']
                    # print(f"rank_zero_info: {rank_zero_info}")
                    # print(f"master_addr: {self._master_addr}, master_port: {self._master_port}")

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def worker_names(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._worker_names

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def from_detached(cls, worker_names=None, ray_cls_with_init=None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        worker_group = cls(resource_pool=None,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           ray_cls_with_init=ray_cls_with_init,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           name_prefix=None,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           worker_names=worker_names)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return worker_group

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def spawn(self, prefix_set):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        spawn to a dictionary of worker groups, each with a subset of method with prefix.

        """

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def _rebind_actor_methods(worker_group, actor_name):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            """
            bind the method with actor_prefix to its original name
            """
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prefix: str = actor_name + '_'
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for method_name in dir(worker_group):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if method_name.startswith(prefix):
                    # only valid when Python >= 3.9
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    original_method_name = method_name.removeprefix(prefix)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    method = getattr(worker_group, method_name)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    setattr(worker_group, original_method_name, method)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_worker_group_dict = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for prefix in prefix_set:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_worker_group = self.from_detached(worker_names=self._worker_names,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  ray_cls_with_init=self.ray_cls_with_init)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _rebind_actor_methods(new_worker_group, prefix)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_worker_group_dict[prefix] = new_worker_group
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return new_worker_group_dict

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ray.get(self.execute_all_async(method_name, **args, **kwargs))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        remote_call = getattr(self._workers[0], method_name)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return remote_call.remote(*args, **kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_all(self, method_name: str, *args, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.execute_all_async(method_name, *args, **kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_all_sync(self, method_name: str, *args, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_all_async(self, method_name: str, *args, **kwargs):
        # 这里我们假设，如果 args 和 kwargs 里面所有的参数都是 list，且所有的 list 长度都与 len(self._workers) 一致的话，我们会把
        # list 中的每一个分别发到对应的 worker 上去
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        length = len(self._workers)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                result = []
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for i in range(length):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    sliced_args = tuple(arg[i] for arg in args)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    remote_call = getattr(self._workers[i], method_name)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return result

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self._workers]

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def master_address(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._master_addr

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def master_port(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._master_port

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def workers(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._workers

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def world_size(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._world_size


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Utilities that enables creating workers inside the same ray.Actor, 
with code written in separate ray.Actors.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from unittest.mock import patch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.decorator import MAGIC_ATTR
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Binds the methods of each worker to the WorkerDict. 
    Note that we only bind public methods that are decorated by register
    """
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for method_name in dir(user_defined_cls):
        # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
        try:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            method = getattr(user_defined_cls, method_name)
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        # 中文注释：下一行处理异常分支，保证错误可控。
        except Exception as e:
            # if it is a property, it will fail because Class doesn't have instance property
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            continue

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hasattr(method, MAGIC_ATTR):

            # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
            def generate_function(name):

                # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return func

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            setattr(func, MAGIC_ATTR, getattr(method, MAGIC_ATTR))
            # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
            try:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                method_name_with_prefix = key + '_' + method_name
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                setattr(cls, method_name_with_prefix, func)
                # print(f'Binding {method_name_with_prefix}')
            # 中文注释：下一行处理异常分支，保证错误可控。
            except Exception as e:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f'Fail to set method_name {method_name}')


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _unwrap_ray_remote(cls):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if hasattr(cls, '__ray_actor_class__'):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cls = cls.__ray_actor_class__
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return cls


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    This function should return a class instance that delegates the calls to every 
    cls in cls_dict
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cls_dict = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    init_args_dict = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    worker_cls = None
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, cls in class_dict.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if worker_cls == None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            worker_cls = cls.cls.__ray_actor_class__.__base__
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert worker_cls == cls.cls.__ray_actor_class__.__base__, \
                'the worker class should be the same when share the same process'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cls_dict[key] = cls.cls
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    # 中文注释：下一行定义类，用于组织相关状态与行为。
    class WorkerDict(worker_cls):

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def __init__(self):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            super().__init__()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.worker_dict = {}
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for key, user_defined_cls in cls_dict.items():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.worker_dict[key] = user_defined_cls(*init_args_dict[key].get('args', ()),
                                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                             **init_args_dict[key].get('kwargs', {}))

    # now monkey-patch the methods from inner class to WorkerDict
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, user_defined_cls in cls_dict.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    remote_cls = ray.remote(WorkerDict)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return remote_cls
