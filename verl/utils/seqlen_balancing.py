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
from typing import List, Tuple, Callable
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import heapq

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import distributed as dist

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import copy


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    # 中文注释：下一行定义类，用于组织相关状态与行为。
    class Set:

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def __init__(self) -> None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.sum = 0
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.items = []

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def add(self, idx: int, val: int):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.items.append((idx, val))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.sum += val

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def merge(self, other):
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for idx, val in other.items:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.items.append((idx, val))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.sum += val

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def __lt__(self, other):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.sum != other.sum:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return self.sum < other.sum
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if len(self.items) != len(other.items):
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return len(self.items) < len(other.items)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.items < other.items

    # 中文注释：下一行定义类，用于组织相关状态与行为。
    class State:

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.k = k
            # sets should always be decreasing order
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.sets = [Set() for _ in range(k)]
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i, (idx, seqlen) in enumerate(items):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.sets[i].add(idx=idx, val=seqlen)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.sets = sorted(self.sets, reverse=True)

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def spread(self):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.sets[0].sum - self.sets[-1].sum

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def get_partitions(self):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            partitions = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(len(self.sets)):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                cur_partition = []
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for idx, _ in self.sets[i].items:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    cur_partition.append(idx)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                partitions.append(cur_partition)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return partitions

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def merge(self, other):
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(self.k):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.sets[i].merge(other.sets[self.k - 1 - i])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.sets = sorted(self.sets, reverse=True)

        # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
        @property
        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def spread(self) -> int:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.sets[0].sum - self.sets[-1].sum

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.spread != other.spread:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return self.spread > other.spread
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.sets[0] > other.sets[0]

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def __repr__(self) -> str:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            repr_str = "["
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(self.k):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if i > 0:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    repr_str += ","
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                repr_str += "{"
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if j > 0:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        repr_str += ","
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    repr_str += str(seqlen)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                repr_str += "}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            repr_str += "]"
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return repr_str

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    states_pq = []
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if equal_size:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            items = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(k_partitions):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seqlen, idx = sorted_seqlen_list[offset + i]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                items.append((idx, seqlen))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seqlen, idx in sorted_seqlen_list:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    # 中文注释：下一行开始循环，直到条件不再满足。
    while len(states_pq) > 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state0 = heapq.heappop(states_pq)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state1 = heapq.heappop(states_pq)
        # merge states
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state0.merge(state1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        heapq.heappush(states_pq, state0)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    final_state = states_pq[0]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    partitions = final_state.get_partitions()
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if equal_size:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, partition in enumerate(partitions):
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(partition) * \
                k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return partitions


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def greedy_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bias = sum(seqlen_list) + 1 if equal_size else 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    partitions = [[] for _ in range(k_partitions)]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    partition_sums = [0 for _ in range(k_partitions)]
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for seqlen, i in sorted_seqlen:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        min_idx = None
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for j in range(k_partitions):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                min_idx = j
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        partitions[min_idx].append(i)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        partition_sums[min_idx] += seqlen
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if equal_size:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, partition in enumerate(partitions):
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(partition) * \
                k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return partitions


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """ get order of seq lengths to make partitions balanced, this is
        used in balacing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_and_sort_partitions(partitions):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seen_idx = set()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sorted_partitions = [None] * k_partitions
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, partition in enumerate(partitions):
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(partition) > 0, f"the {i}-th partition is empty"
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for idx in partition:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seen_idx.add(idx)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sorted_partitions[i] = sorted(partition)
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert seen_idx == set(range(len(seqlen_list)))
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return sorted_partitions

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _check_and_sort_partitions(partitions)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def log_seqlen_unbalance(seqlen_list: List[int], partitions: List[List[int]], prefix):
    # add some metrics of seqlen sum on dp ranks
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size = len(seqlen_list) // k_partition
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_sum_seqlen = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_sum_seqlen = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_sum_seqlen = 0
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for offset in range(0, len(seqlen_list), batch_size):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cur_sum_seqlen = sum(seqlen_list[offset:offset + batch_size])
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            min_sum_seqlen = cur_sum_seqlen
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_sum_seqlen = cur_sum_seqlen
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_sum_seqlen += cur_sum_seqlen

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    balanced_sum_seqlen_list = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for partition in partitions:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'{prefix}/min': min_sum_seqlen,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'{prefix}/max': max_sum_seqlen,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'{prefix}/minmax_diff': max_sum_seqlen - min_sum_seqlen,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'{prefix}/balanced_min': min_sum_seqlen_balanced,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'{prefix}/balanced_max': max_sum_seqlen_balanced,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        f'{prefix}/mean': total_sum_seqlen / len(partitions)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def ceildiv(a, b):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return -(a // -b)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def rearrange_micro_batches(batch: TensorDict, max_token_len, dp_group=None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Split the batch into a list of micro_batches, where the max_token_len is smaller than max_token_len
    and the number of valid tokens in each micro batch is well balanced.
    """
    # this is per local micro_bsz
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_seq_len = batch['attention_mask'].shape[-1]
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert max_token_len >= max_seq_len, \
        f'max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}'

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seq_len_effective: torch.Tensor = batch['attention_mask'].sum(dim=1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_seqlen = seq_len_effective.sum().item()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_micro_batches = ceildiv(total_seqlen, max_token_len)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if dist.is_initialized():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_micro_batches = torch.tensor([num_micro_batches], device='cuda')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_micro_batches = num_micro_batches.cpu().item()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seq_len_effective = seq_len_effective.tolist()
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert num_micro_batches <= len(seq_len_effective)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    micro_batches = []

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for partition in micro_bsz_idx:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        curr_micro_batch = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx in partition:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            curr_micro_batch.append(batch[idx:idx + 1])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        curr_micro_batch = torch.cat(curr_micro_batch)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batches.append(curr_micro_batch)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return micro_batches, micro_bsz_idx


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_reverse_idx(idx_map):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    reverse_idx_map = copy.deepcopy(idx_map)

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i, idx in enumerate(idx_map):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reverse_idx_map[idx] = i

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return reverse_idx_map
