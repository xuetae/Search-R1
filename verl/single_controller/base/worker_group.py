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
the class of WorkerGroup
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import threading
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import signal
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Any, Callable, Dict

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch, get_predefined_dispatch_fn, get_predefined_execute_fn


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ResourcePool:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, process_on_nodes=None, max_collocate_count: int = 10, n_gpus_per_node=8) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if process_on_nodes is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            process_on_nodes = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._store = process_on_nodes
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_collocate_count = max_collocate_count
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.n_gpus_per_node = n_gpus_per_node  # this is left for future huawei GPU that contains 16 GPUs per node

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_node(self, process_count):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._store.append(process_count)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def world_size(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return sum(self._store)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self) -> Any:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._store

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def store(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._store

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def local_world_size_list(self) -> List[int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nested_local_world_size_list = [
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            [local_world_size for _ in range(local_world_size)] for local_world_size in self._store
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return [item for row in nested_local_world_size_list for item in row]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def local_rank_list(self) -> List[int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nested_local_rank_list = [[i for i in range(local_world_size)] for local_world_size in self._store]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return [item for row in nested_local_rank_list for item in row]


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ClassWithInitArgs:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    This class stores a class constructor and the args/kwargs to construct the class.
    It is used to instantiate the remote class.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, cls, *args, **kwargs) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cls = cls
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.args = args
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.kwargs = kwargs

    # def add_arg(self, arg):
    #     self.args += (arg,)

    # def add_kwarg(self, key, value):
    #     self.kwargs[key] = value

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self) -> Any:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.cls(*self.args, **self.kwargs)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def check_workers_alive(workers: List, is_alive: Callable, gap_time: float = 1) -> None:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import time
    # 中文注释：下一行开始循环，直到条件不再满足。
    while True:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for worker in workers:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not is_alive(worker):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logging.warning(f"worker {worker} is not alive" + " sending signal to main thread")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                signal.raise_signal(signal.SIGABRT)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        time.sleep(gap_time)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class WorkerGroup:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, resource_pool: ResourcePool, **kwargs) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_init_with_detached_workers = True if resource_pool is None else False

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if resource_pool is not None:
            # handle the case when WorkGroup is attached to an existing one
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._procecss_dispatch_config = resource_pool()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._procecss_dispatch_config = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._workers = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._worker_names = []

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._master_addr = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._master_port = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._checker_thread: threading.Thread = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _is_worker_alive(self, worker):
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError(f"WorkerGroup._is_worker_alive called, should be implemented in derived class.")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _block_until_all_workers_alive(self) -> None:
        # 中文注释：下一行开始循环，直到条件不再满足。
        while True:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_state = [self._is_worker_alive(worker) for worker in self._workers]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if False in all_state:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                time.sleep(1)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                break

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def start_worker_aliveness_check(self, every_n_seconds=1) -> None:
        # before starting checking worker aliveness, make sure all workers are already alive
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._block_until_all_workers_alive()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._checker_thread = threading.Thread(target=check_workers_alive,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                args=(self._workers, self._is_worker_alive, every_n_seconds))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._checker_thread.start()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def world_size(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return len(self._workers)

    # execute_all_async and execute_rank_zero_async should be implemented by RayWorkerGroup, TorchRPCWorkerGroup,
    # MegatronWorkerGroup, XperfWorkerGroup should skip

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _bind_worker_method(self, user_defined_cls, func_generator):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Bind the worker method to the WorkerGroup
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
                # this method is decorated by register
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                attribute = getattr(method, MAGIC_ATTR)
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert isinstance(attribute, Dict), f'attribute must be a dictionary. Got {type(attribute)}'
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert 'dispatch_mode' in attribute, f'attribute must contain dispatch_mode in its key'

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dispatch_mode = attribute['dispatch_mode']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                execute_mode = attribute['execute_mode']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                blocking = attribute['blocking']

                # get dispatch fn
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if isinstance(dispatch_mode, Dispatch):
                    # get default dispatch fn
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dispatch_fn = fn['dispatch_fn']
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    collect_fn = fn['collect_fn']
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                    assert isinstance(dispatch_mode, dict)
                    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                    assert 'dispatch_fn' in dispatch_mode
                    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                    assert 'collect_fn' in dispatch_mode
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dispatch_fn = dispatch_mode['dispatch_fn']
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    collect_fn = dispatch_mode['collect_fn']

                # get execute_fn_name
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                wg_execute_fn_name = execute_mode['execute_fn_name']

                # get execute_fn from string
                # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
                try:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    execute_fn = getattr(self, wg_execute_fn_name)
                    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                    assert callable(execute_fn), 'execute_fn must be callable'
                # 中文注释：下一行处理异常分支，保证错误可控。
                except Exception as e:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    print(f'execute_fn {wg_execute_fn_name} is invalid')
                    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                    raise

                # bind a new method to the RayWorkerGroup
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                func = func_generator(self,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      method_name,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      dispatch_fn=dispatch_fn,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      collect_fn=collect_fn,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      execute_fn=execute_fn,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      blocking=blocking)

                # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
                try:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    setattr(self, method_name, func)
                # 中文注释：下一行处理异常分支，保证错误可控。
                except Exception as e:
                    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                    raise ValueError(f'Fail to set method_name {method_name}')
