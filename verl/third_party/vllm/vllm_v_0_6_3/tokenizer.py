# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/tokenizer_group/tokenizer_group.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PreTrainedTokenizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import LRUCache


# 中文注释：下一行定义类，用于组织相关状态与行为。
class TokenizerGroup(TokenizerGroup):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """A group of tokenizers that can be used for LoRA adapters."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, tokenizer: PreTrainedTokenizer, enable_lora: bool, max_num_seqs: int,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 max_input_length: Optional[int]):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.enable_lora = enable_lora
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_input_length = max_input_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_tokenizers = LRUCache[PreTrainedTokenizer](capacity=max_num_seqs) if enable_lora else None

    # FIXME(sgm): for simplicity, we assign the special token here
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pad_token_id(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.tokenizer.pad_token_id

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def eos_token_id(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.tokenizer.eos_token_id
