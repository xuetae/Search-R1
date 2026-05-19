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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Iterable, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn.functional as F
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import logprobs_from_logits
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from ..base import BaseRollout

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['NativeRollout']


# 中文注释：下一行定义类，用于组织相关状态与行为。
class NaiveRollout(BaseRollout):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, module: nn.Module, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """A naive rollout. It requires the module to be compatible with huggingface APIs. That is:
        The module should define __call__ to receive input_ids, attention_mask and position_ids.
        It outputs a structure that contains logits field.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module = module

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Generate sequences"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_token_id = prompts.meta_info['eos_token_id']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = idx.size(0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_length = idx.size(1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.eval()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prev_attention_mask = torch.ones(size=(batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for _ in range(self.config.response_length):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            # we use huggingface APIs here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.module(input_ids=idx_cond, attention_mask=attention_mask, position_ids=position_ids)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = output.logits
            # pluck the logits at the final step and scale by desired temperature
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = logits[:, -1, :] / self.config.temperature  # (bs, vocab_size)
            # optionally crop the logits to only the top k options
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.top_k is not None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.do_sample:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                idx_next = torch.multinomial(probs, num_samples=1)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = torch.cat((attention_mask, prev_attention_mask), dim=-1)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prev_attention_mask = torch.logical_and(idx_next != eos_token_id, prev_attention_mask.bool())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prev_attention_mask.to(attention_mask.dtype)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)

            # append sampled index to the running sequence and continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            idx = torch.cat((idx, idx_next), dim=1)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits_lst.append(logits)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = torch.stack(logits_lst, dim=1)  # (bs, response_length, vocab_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompts = idx[:, :prompt_length]  # (bs, prompt_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response = idx[:, prompt_length:]  # (bs, response_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_probs = logprobs_from_logits(logits=logits, labels=response)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = TensorDict(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'input_ids': prompts,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'responses': response,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'sequences': idx,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'old_log_probs': log_probs,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'attention_mask': attention_mask,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'position_ids': position_ids,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            },
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size=batch_size)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.train()

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(batch=batch)
