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
from typing import List, Optional, Tuple, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast)

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import make_async, LRUCache
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.transformers_utils.tokenizers import *


# 中文注释：下一行定义类，用于组织相关状态与行为。
class TokenizerGroup:
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
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if enable_lora:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_tokenizers = LRUCache(capacity=max_num_seqs)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_tokenizers = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def encode(self,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               prompt: str,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               request_id: Optional[str] = None,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               lora_request: Optional[LoRARequest] = None) -> List[int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer = self.get_lora_tokenizer(lora_request)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return tokenizer.encode(prompt)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    async def encode_async(self,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           prompt: str,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           request_id: Optional[str] = None,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           lora_request: Optional[LoRARequest] = None) -> List[int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return tokenizer.encode(prompt)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_lora_tokenizer(self, lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not lora_request or not self.enable_lora:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.tokenizer
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if lora_request.lora_int_id not in self.lora_tokenizers:
            # TODO(sgm): the lora tokenizer is also passed, but may be different
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer = self.tokenizer
            # tokenizer = (get_lora_tokenizer(
            #     lora_request, **self.tokenizer_config) or self.tokenizer)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return tokenizer
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.lora_tokenizers.get(lora_request.lora_int_id)

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
