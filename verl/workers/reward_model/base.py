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
The base class for reward model
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from abc import ABC, abstractmethod

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BasePPORewardModel(ABC):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_reward(self, data: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Computing reward given input_ids. The transformers should output a tensor with shape
           [batch_size, sequence_length], and the value at [EOS] mask should be gathered.

        Args:
            data: must contain keys "input_ids", "attention_mask" and "position_ids".
                - input_ids: [batch_size, sequence_length]
                - attention_mask: [batch_size, sequence_length]
                - position_ids: [batch_size, sequence_length]

        Returns: a data pass protocol containing "reward". Only the [EOS] position contains the reward.
            Other position should have zero reward. Note that this may change in the future if we use
            dense reward. So, we leave the interface for general case.
            - reward: [batch_size, sequence_length].

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass
