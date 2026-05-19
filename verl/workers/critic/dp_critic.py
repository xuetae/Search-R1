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
Implement a multiprocess PPOCritic
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import itertools
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Iterable

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn, optim

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo import core_algos
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.critic import BasePPOCritic
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.py_functional import append_to_dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import masked_mean
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['DataParallelPPOCritic']


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DataParallelPPOCritic(BasePPOCritic):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config=config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_module = critic_module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_optimizer = critic_optimizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Critic use_remove_padding={self.use_remove_padding}')

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_micro_batch(self, micro_batch):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = micro_batch['responses'].size(-1)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = micro_batch['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch, seqlen = input_ids.shape
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

                # pad and slice the inputs if sp > 1
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.ulysses_sequence_parallel_size > 1:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            attention_mask=None,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            position_ids=position_ids_rmpad,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            use_cache=False)  # prevent model thinks we are generating
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values_rmpad = output.logits
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.ulysses_sequence_parallel_size > 1:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           gather_dim=0,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           unpad_dim=0,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           padding_size=pad_size)

                # pad it back
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = values[:, -response_length - 1:-1]
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.critic_module(input_ids=input_ids,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            attention_mask=attention_mask,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            position_ids=position_ids,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            use_cache=False)  # prevent model thinks we are generating
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = output.logits
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = values[:, -response_length - 1:-1].squeeze(-1)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return values

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _optimizer_step(self):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.config.grad_clip is not None

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(self.critic_module, FSDP):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_optimizer.step()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return grad_norm

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_values(self, data: DataProto) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_module.eval()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batch_size = data.meta_info['micro_batch_size']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = data.select(batch_keys=select_keys).batch
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

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
        values_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for micro_batch in micro_batches:
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.no_grad():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = self._forward_micro_batch(micro_batch)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values_lst.append(values)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        values = torch.concat(values_lst, dim=0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses = data.batch['responses']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = responses.size(1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        values = values * attention_mask[:, -response_length - 1:-1]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_dynamic_bsz:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            indices = list(itertools.chain.from_iterable(indices))
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values = values[revert_indices]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return values

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_module.train()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics = {}

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dataloader = batch.split(self.config.ppo_mini_batch_size)

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
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_optimizer.zero_grad()

            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for data in micro_batches:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = data.cuda()  # critic device is cpu when using offload
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids = data['input_ids']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses = data['responses']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                attention_mask = data['attention_mask']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids = data['position_ids']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = data['values']
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                returns = data['returns']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                response_length = responses.size(1)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                eos_mask = attention_mask[:, -response_length - 1:-1]

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                vpreds = self._forward_micro_batch(data)

                # assert not torch.any(torch.isnan(vpreds)).item()

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                     values=values,
                                                                     # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                                                     returns=returns,
                                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                     eos_mask=eos_mask,
                                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                     cliprange_value=self.config.cliprange_value)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                loss = vf_loss / self.gradient_accumulation
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                loss.backward()

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'critic/vf_loss': vf_loss.detach().item(),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                append_to_dict(metrics, data)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_norm = self._optimizer_step()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = {'critic/grad_norm': grad_norm.detach().item()}
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            append_to_dict(metrics, data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_optimizer.zero_grad()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return metrics
