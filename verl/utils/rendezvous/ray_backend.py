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
import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from cupy.cuda.nccl import NcclCommunicator, get_unique_id

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from ray.util import list_named_actors


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@ray.remote
# 中文注释：下一行定义类，用于组织相关状态与行为。
class NCCLIDStore:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, nccl_id):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._nccl_id = nccl_id

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._nccl_id


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_nccl_id_store_by_name(name):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    all_actors = list_named_actors(all_namespaces=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(matched_actors) == 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor = matched_actors[0]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ray.get_actor(**actor)
    # 中文注释：下一行继续判断其他条件分支。
    elif len(matched_actors) > 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logging.warning(f"multiple actors with same name found: {matched_actors}")
    # 中文注释：下一行继续判断其他条件分支。
    elif len(matched_actors) == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logging.info(f"failed to get any actor named {name}")
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return None


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def create_nccl_communicator_in_ray(rank: int,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    world_size: int,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    group_name: str,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    max_retries: int = 100,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    interval_s: int = 5):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if rank == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nccl_id = get_unique_id()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert ray.get(nccl_id_store.get.remote()) == nccl_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        communicator = NcclCommunicator(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ndev=world_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            commId=nccl_id,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank=0,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return communicator
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(max_retries):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            nccl_id_store = get_nccl_id_store_by_name(group_name)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if nccl_id_store is not None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logging.info(f"nccl_id_store {group_name} got")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                nccl_id = ray.get(nccl_id_store.get.remote())
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logging.info(f"nccl id for {group_name} got: {nccl_id}")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                communicator = NcclCommunicator(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    ndev=world_size,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    commId=nccl_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    rank=rank,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return communicator
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logging.info(f"failed to get nccl_id for {i+1} time, sleep for {interval_s} seconds")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time.sleep(interval_s)
