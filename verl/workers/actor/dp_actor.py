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
Single Process Actor
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import itertools
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Iterable, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo import core_algos
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.actor import BasePPOActor
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.py_functional import append_to_dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.torch_functional as verl_F

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['DataParallelPPOActor']


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DataParallelPPOActor(BasePPOActor):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_module: nn.Module,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_optimizer: torch.optim.Optimizer = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """When optimizer is None, it is Reference Policy"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_module = actor_module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_optimizer = actor_optimizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = micro_batch['responses'].size(-1)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = micro_batch['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size, seqlen = input_ids.shape
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = micro_batch['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = micro_batch['position_ids']

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.use_remove_padding:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.use_ulysses_sp:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                                self.ulysses_sequence_parallel_size)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           attention_mask=None,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           position_ids=position_ids_rmpad,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           use_cache=False)  # prevent model thinks we are generating
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits_rmpad.div_(temperature)

                # compute entropy
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            gather_dim=0,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            unpad_dim=0,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         indices=indices,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         batch=batch_size,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         seqlen=seqlen)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           indices=indices,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           batch=batch_size,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           seqlen=seqlen)

                # only return response part:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:  # not using rmpad and no ulysses sp
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.actor_module(input_ids=input_ids,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           attention_mask=attention_mask,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           position_ids=position_ids,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           use_cache=False)  # prevent model thinks we are generating
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits = output.logits
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits.div_(temperature)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return entropy, log_probs

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _optimizer_step(self):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.config.grad_clip is not None

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(self.actor_module, FSDP):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_optimizer.step()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return grad_norm

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_module.eval()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batch_size = data.meta_info['micro_batch_size']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = data.select(batch_keys=select_keys).batch

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_dynamic_bsz:
            # split using dynamic bsz
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            micro_batches = batch.split(micro_batch_size)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_probs_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for micro_batch in micro_batches:
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.no_grad():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_probs_lst.append(log_probs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_probs = torch.concat(log_probs_lst, dim=0)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_dynamic_bsz:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            indices = list(itertools.chain.from_iterable(indices))
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_probs = log_probs[revert_indices]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return log_probs

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_module.train()

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.state_masking:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            select_keys.append('loss_mask')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.use_kl_loss:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            select_keys.append('ref_log_prob')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mini_batch = data
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.use_dynamic_bsz:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # split batch into micro_batches
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_optimizer.zero_grad()

            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for data in micro_batches:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = data.cuda()  # actor device is cpu when using offload
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses = data['responses']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                response_length = responses.size(1)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                attention_mask = data['attention_mask']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                response_mask = attention_mask[:, -response_length:]
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.config.state_masking:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    response_mask = data['loss_mask']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                old_log_prob = data['old_log_probs']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                advantages = data['advantages']

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                clip_ratio = self.config.clip_ratio
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                              log_prob=log_prob,
                                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                              advantages=advantages,
                                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                              eos_mask=response_mask,
                                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.config.use_kl_loss:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                ref_logprob=ref_log_prob,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                kl_penalty=self.config.kl_loss_type)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    kl_loss = masked_mean(kld, response_mask)

                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                loss = policy_loss / self.gradient_accumulation
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                loss.backward()

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'actor/pg_loss': pg_loss.detach().item(),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'actor/ppo_kl': ppo_kl.detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                append_to_dict(metrics, data)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_norm = self._optimizer_step()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = {'actor/grad_norm': grad_norm.detach().item()}
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            append_to_dict(metrics, data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_optimizer.zero_grad()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return metrics
