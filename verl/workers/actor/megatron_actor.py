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
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from functools import partial
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Iterable, Dict

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# from megatron import get_args
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.optimizer import DistributedOptimizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.optimizer_config import OptimizerConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.pipeline_parallel import get_forward_backward_func
# from megatron.core.optimizer import DistributedOptimizer

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import OmegaConf
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.tensor_parallel import vocab_parallel_compute_entropy_loss, vocab_parallel_log_probs_from_logits
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo import core_algos
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.actor import BasePPOActor
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.py_functional import append_to_dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import logprobs_from_logits, broadcast_dict_tensor, split_dict_tensor_into_batches

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['MegatronPPOActor']


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronPPOActor(BasePPOActor):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config, model_config, megatron_config: ModelParallelConfig, actor_module: nn.ModuleList,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 actor_optimizer: DistributedOptimizer, actor_optimizer_config: OptimizerConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size``: minibatch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            megatron_config (OmegaConf): megatron configuration. It must contains

                ``sequence_parallel_enabled``: whether the sequence parallel is enabled.

                ``param_dtype``: the dtype of the parameters.

                ``virtual_pipeline_model_parallel_size``: virtual pipeline model parallel size. a.k.a number of chunks in each pp stage.
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation. Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron. It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> def megatron_actor_model_provider(pre_process, post_process):
        >>>     vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        >>>     parallel_model = ParallelMistralForCausalLMRmPadPP(config=actor_model_config,
        >>>                                                        megatron_config=megatron_config,
        >>>                                                        pre_process=pre_process,
        >>>                                                        post_process=post_process).cuda()
        >>>     return parallel_model
        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          megatron_config=megatron_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # self.megatron_args = get_args()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_module = actor_module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_optimizer_config = actor_optimizer_config

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
            DataProto: torch.Tensor: the log_prob tensor
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.contiguous()

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def compute_logprobs_fn(output, data):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response = data['responses']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_length = response.size(1)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = output['logits']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = logits[:, -response_length - 1:-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_probs = vocab_parallel_log_probs_from_logits(logits, response)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return {'log_probs': log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be handled by user outside
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        recompute_old_log_prob = self.config.get('recompute_old_log_prob', True)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if recompute_old_log_prob or 'old_log_probs' not in data.batch.keys():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch = data.select(batch_keys=select_keys).batch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = batch['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size = input_ids.size(0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response = batch['responses']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_length = response.size(1)
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.no_grad():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.forward_backward_batch(data, forward_only=True, post_process_fn=compute_logprobs_fn)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    log_probs = torch.cat([o['log_probs'] for o in output], dim=0)  # (bs, seq_size)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    log_probs = log_probs.to(torch.float32)
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    log_probs = torch.empty(size=(batch_size, response_length),
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            dtype=torch.float32,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            device=input_ids.device)

                # broadcast across pp ranks
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                torch.distributed.broadcast(tensor=log_probs,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            src=mpu.get_pipeline_model_parallel_last_rank(),
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            group=mpu.get_pipeline_model_parallel_group(),
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            async_op=False)

        # add empty cache after each compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return log_probs

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.select(batch_keys=select_keys)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data.make_iterator(mini_batch_size=self.config.ppo_mini_batch_size,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  epochs=self.config.ppo_epochs,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  dataloader_kwargs={'shuffle': self.config.shuffle})

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward_backward_batch(self, data: DataProto, forward_only=False, post_process_fn=None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        broadcast_dict_tensor(data.batch,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              group=mpu.get_pipeline_model_parallel_group())
        # split into micro-batches
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if data.meta_info.get('micro_batch_size', None) is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size = data.meta_info['micro_batch_size']
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size = self.config.ppo_micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)
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
        n_micro_batch = len(batches)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_len = batches[0]['input_ids'].shape[1]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        forward_backward_func = get_forward_backward_func()

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def loss_func(output, data, meta_info):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if forward_only:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if post_process_fn is None:
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return 1.0, {'logits': output.logits}
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return 1.0, post_process_fn(output, data)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses = data['responses']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_length = responses.size(1)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = data['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_mask = attention_mask[:, -response_length:]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            old_log_prob = data['old_log_probs']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            advantages = data['advantages']

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            clip_ratio = meta_info['clip_ratio']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            entropy_coeff = meta_info['entropy_coeff']

            # compute policy loss
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = output.logits
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits = logits[:, -response_length - 1:-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_prob = vocab_parallel_log_probs_from_logits(logits, responses)
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
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            entropy_loss = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            policy_loss = pg_loss - entropy_loss * entropy_coeff
            # return loss and stats
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            stats = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'actor/entropy_loss': entropy_loss.detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'actor/pg_loss': pg_loss.detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'actor/ppo_kl': ppo_kl.detach().item()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return policy_loss, stats

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
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if forward_only:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                meta_info = None
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                meta_info = {'clip_ratio': self.config.clip_ratio, 'entropy_coeff': self.config.entropy_coeff}
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_generator = make_batch_generator(batches, vpp_size=len(self.actor_module))

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
                model=self.actor_module,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_microbatches=n_micro_batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_shapes=input_shapes,  # must set for flash-attn sequence packing
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_length=batch_size * seq_len,  # no use when input_shapes was set
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
                model=self.actor_module,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_microbatches=n_micro_batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_length=batch_size * seq_len,  # in use for pp = 1
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
    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for data in dataloader:
            # data = data.batch.to(self.actor_module.device)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                chunk.zero_grad_buffer(zero_buffer=(not self.actor_optimizer_config.use_distributed_optimizer))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metric_micro_batch = self.forward_backward_batch(data)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for metric in metric_micro_batch:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step(
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
