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
Trajectory tracker can be inserted into code to save the intermediate results.
The results will be dump to hdfs for offline comparison.
Each process will have a client that first move all the tensors to CPU
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.hdfs_io import makedirs, copy
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import io
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import tempfile

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from collections import deque

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
remote_copy = ray.remote(copy)


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@ray.remote
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def save_to_hdfs(data: io.BytesIO, name, hdfs_dir, verbose):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    filename = name + '.pth'
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with tempfile.TemporaryDirectory() as tmpdirname:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_filepath = os.path.join(tmpdirname, filename)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with open(local_filepath, 'wb') as f:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            f.write(data.getbuffer())
        # upload to hdfs

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if verbose:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Saving {local_filepath} to {hdfs_dir}')
        # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
        try:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            copy(local_filepath, hdfs_dir)
        # 中文注释：下一行处理异常分支，保证错误可控。
        except Exception as e:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(e)


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@ray.remote
# 中文注释：下一行定义类，用于组织相关状态与行为。
class TrajectoryTracker():

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, hdfs_dir, verbose) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hdfs_dir = hdfs_dir
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        makedirs(hdfs_dir)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.verbose = verbose

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.handle = deque()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def dump(self, data: io.BytesIO, name):
        # get a temp file and write to it
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.handle.append(save_to_hdfs.remote(data, name, self.hdfs_dir, self.verbose))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def wait_for_hdfs(self):
        # 中文注释：下一行开始循环，直到条件不再满足。
        while len(self.handle) != 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            future = self.handle.popleft()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ray.get(future)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def dump_data(data, name):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enable = os.getenv('VERL_ENABLE_TRACKER', '0') == '1'
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not enable:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    buffer = io.BytesIO()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.save(data, buffer)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tracker = get_trajectory_tracker()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray.get(tracker.dump.remote(buffer, name))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_trajectory_tracker():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hdfs_dir = os.getenv('VERL_TRACKER_HDFS_DIR', default=None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    verbose = os.getenv('VERL_TRACKER_VERBOSE', default='0') == '1'
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert hdfs_dir is not None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tracker = TrajectoryTracker.options(name="global_tracker", get_if_exists=True,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        lifetime="detached").remote(hdfs_dir, verbose)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return tracker


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # testing
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    os.environ['VERL_ENABLE_TRACKER'] = '1'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    os.environ['VERL_TRACKER_HDFS_DIR'] = '~/debug/test'

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @ray.remote
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def process(iter):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = {'obs': torch.randn(10, 20)}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dump_data(data, f'process_{iter}_obs')

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray.init()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_lst = []

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(10):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_lst.append(process.remote(i))

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    out = ray.get(output_lst)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tracker = get_trajectory_tracker()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray.get(tracker.wait_for_hdfs.remote())
