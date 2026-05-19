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


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MemoryBuffer:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, numel, numel_padded, dtype):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.numel = numel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.numel_padded = numel_padded
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dtype = dtype
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.data = torch.zeros(self.numel_padded,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                dtype=self.dtype,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                device=torch.cuda.current_device(),
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                requires_grad=False)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def zero(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Reset the buffer to zero."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.data.zero_()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get(self, shape, start_index):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        end_index = start_index + shape.numel()
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = self.data[start_index:end_index]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_tensor = buffer_tensor.view(shape)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return buffer_tensor
