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
from functools import partial
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Iterable

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import OmegaConf
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo import core_algos
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.critic import BasePPOCritic
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.py_functional import append_to_dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_dtypes import PrecisionType
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import masked_mean, broadcast_dict_tensor, split_dict_tensor_into_batches
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import sequence_parallel as sp_utils
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.optimizer_config import OptimizerConfig

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.optimizer import DistributedOptimizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.pipeline_parallel import get_forward_backward_func


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronPPOCritic(BasePPOCritic):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config, model_config, megatron_config, critic_module: nn.ModuleList,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 critic_optimizer: DistributedOptimizer, critic_optimizer_config: OptimizerConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config=config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_module = critic_module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_optimizer = critic_optimizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_optimizer_config = critic_optimizer_config

        # we create a separate nametuple for optimizer step so that global args won't affect it.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.optimizer_step_args = OmegaConf.create({
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'skip_grad': None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'overlap_dp_param_comm': False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'overlap_dp_grad_comm': False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'gradient_accumulation_steps': 1,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'sequence_parallel': self.megatron_config.sequence_parallel,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'DDP_impl': 'local',
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'layernorm_allreduce_bucket_threshold': 0,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_split_rank': None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'reduce_grads_use_alltoall': False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        })

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.kl_ctrl.type == 'fixed':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=self.config.kl_ctrl.kl_coef)
        # 中文注释：下一行继续判断其他条件分支。
        elif self.config.kl_ctrl.type == 'adaptive':
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert self.config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {self.config.kl_ctrl.horizon}'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=self.config.kl_ctrl.kl_coef,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           target_kl=self.config.kl_ctrl.target_kl,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           horizon=self.config.kl_ctrl.horizon)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_values(self, data: DataProto) -> DataProto:
        # data.batch = data.batch.to(self.critic_module.module.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses = data.batch['responses']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = responses.size(1)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.no_grad():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.forward_backward_batch(data=data, forward_only=True)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = torch.cat([o['vpreds'] for o in output], dim=0)  # (bs, seq_size, vocal_size)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = values.to(torch.float32)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            # each tp ranks should contain the same value
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values = values * attention_mask
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values = values[:, -response_length - 1:-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values = values.contiguous()

            # sync among pp ranks
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.broadcast(tensor=values,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        group=mpu.get_pipeline_model_parallel_group())

        # add empty cache after each compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return values

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.select(batch_keys=select_keys)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  epochs=self.config.ppo_epochs,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward_backward_batch(self, data: DataProto, forward_only=False):
        # broadcast from last pp rank to all other pp ranks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.contiguous()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        broadcast_dict_tensor(data.batch,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batches = split_dict_tensor_into_batches(data.batch, batch_size=self.config.ppo_micro_batch_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        n_micro_batch = len(batches)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_len = batches[0]['input_ids'].shape[1]

        # compute input shapes for pp stages
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_shapes = compute_transformers_input_shapes(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batches,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info={
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'sequence_parallel': self.megatron_config.sequence_parallel,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'hidden_size': self.model_config.hidden_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        forward_backward_func = get_forward_backward_func()

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def loss_func(output, data, meta_info):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if forward_only:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return 1.0, {'vpreds': output.logits}

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses = data['responses']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = data['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values = data['values']
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            returns = data['returns']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_length = responses.size(1)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            eos_mask = attention_mask[:, -response_length:]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cliprange_value = self.config.cliprange_value

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            vpreds = output.logits  # (bs, sequence_length)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            vpreds = vpreds[:, -response_length - 1:-1]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                 values=values,
                                                                 # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                                                 returns=returns,
                                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                 eos_mask=eos_mask,
                                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                 cliprange_value=cliprange_value)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            stats = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'critic/vf_loss': vf_loss.detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }

            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return vf_loss, stats

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def forward_step(batch_iter, model):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch = next(batch_iter)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = batch['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = batch['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = batch['position_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return output, partial(loss_func, data=batch, meta_info={})

        # batch should be a list of batches inside micro-batches
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_generator = make_batch_generator(batches, vpp_size=len(self.critic_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            losses_reduced = forward_backward_func(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_step_func=forward_step,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_iterator=batch_generator,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model=self.critic_module,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_microbatches=n_micro_batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_shapes=input_shapes,  # must set for flash-attn sequence packing
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_length=self.config.ppo_micro_batch_size * seq_len,  # no use when input_shapes was set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_size=self.model_config.hidden_size,  # no use when input_shapes was set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batch_size=1,  # no use when input_shapes was set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_only=forward_only,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            losses_reduced = forward_backward_func(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_step_func=forward_step,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_iterator=batch_generator,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model=self.critic_module,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_microbatches=n_micro_batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_length=self.config.ppo_micro_batch_size * seq_len,  # in use for pp = 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_size=self.model_config.hidden_size,  # in use for pp = 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batch_size=1,  # in use for pp = 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_only=forward_only,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # loss_reduces contains the stats returned from loss_func
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return losses_reduced

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_critic(self, dataloader: Iterable[DataProto]):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics = {}

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for data in dataloader:
            # data = data.batch.to(self.critic_module.device)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for chunk in self.critic_module:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk.zero_grad_buffer(zero_buffer=(not self.critic_optimizer_config.use_distributed_optimizer))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metric_micro_batch = self.forward_backward_batch(data)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.megatron_config, self.megatron_config.timers)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pass
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise NotImplementedError

            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for metric in metric_micro_batch:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        # add empty cache after each compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return metrics
