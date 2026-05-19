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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import uuid
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from contextlib import contextmanager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass, field
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from enum import Enum
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pprint import pprint
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Type, Dict

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from collections import defaultdict

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from codetiming import Timer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import OmegaConf, open_dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base import Worker
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.ray.base import create_colocated_worker_cls
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.ppo import core_algos
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
WorkerType = Type[Worker]


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Role(Enum):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    Actor = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    Rollout = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ActorRollout = 2
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    Critic = 3
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    RefPolicy = 4
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    RewardModel = 5
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ActorRolloutRef = 6


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class ResourcePoolManager:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resource_pool_spec: dict[str, list[int]]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mapping: dict[Role, str]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def create_resource_pool(self):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            use_gpu=True,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            max_colocate_count=1,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            name_prefix=resource_pool_name)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.resource_pool_dict[resource_pool_name] = resource_pool

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_resource_pool(self, role: Role) -> RayResourcePool:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Get the resource pool of the worker_cls"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.resource_pool_dict[self.mapping[role]]


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import masked_mean


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    responses = data.batch['responses']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_length = responses.size(1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    token_level_scores = data.batch['token_level_scores']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size = data.batch.batch_size[0]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if 'ref_log_prob' in data.batch.keys():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kld = kld * response_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        beta = kl_ctrl.value
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        beta = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    token_level_rewards = token_level_scores - beta * kld

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    data.batch['token_level_rewards'] = token_level_rewards

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data, metrics


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if adv_estimator == 'gae':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        values = data.batch['values']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses = data.batch['responses']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = responses.size(-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_mask = attention_mask[:, -response_length:]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_rewards = data.batch['token_level_rewards']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                      values=values,
                                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                      eos_mask=response_mask,
                                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                      gamma=gamma,
                                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                      lam=lam)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['advantages'] = advantages
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['returns'] = returns
    # 中文注释：下一行继续判断其他条件分支。
    elif adv_estimator == 'grpo':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_rewards = data.batch['token_level_rewards']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        index = data.non_tensor_batch['uid']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses = data.batch['responses']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = responses.size(-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_mask = attention_mask[:, -response_length:]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                        eos_mask=response_mask,
                                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                        index=index)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['advantages'] = advantages
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['returns'] = returns
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def reduce_metrics(metrics: dict):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in metrics.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics[key] = np.mean(val)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return metrics


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _compute_response_info(batch):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_length = batch.batch['responses'].shape[-1]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt_length = prompt_mask.sum(-1).float()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dict(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_mask=response_mask,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_length=prompt_length,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length=response_length,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    advantages = batch.batch['advantages']
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    returns = batch.batch['returns']

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_response_length = batch.batch['responses'].shape[-1]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_prompt_length = prompt_mask.size(-1)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_info = _compute_response_info(batch)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt_length = response_info['prompt_length']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_length = response_info['response_length']

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    valid_adv = torch.masked_select(advantages, response_mask)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    valid_returns = torch.masked_select(returns, response_mask)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if use_critic:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        values = batch.batch['values']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        valid_values = torch.masked_select(values, response_mask)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return_diff_var = torch.var(valid_returns - valid_values)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return_var = torch.var(valid_returns)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    metrics = {
        # score
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/score/mean':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(sequence_score).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/score/max':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.max(sequence_score).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/score/min':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.min(sequence_score).detach().item(),
        # reward
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/rewards/mean':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(sequence_reward).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/rewards/max':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.max(sequence_reward).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/rewards/min':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.min(sequence_reward).detach().item(),
        # adv
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/advantages/mean':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(valid_adv).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/advantages/max':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.max(valid_adv).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/advantages/min':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.min(valid_adv).detach().item(),
        # returns
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/returns/mean':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(valid_returns).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/returns/max':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.max(valid_returns).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'critic/returns/min':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.min(valid_returns).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **({
            # values
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'critic/values/max': torch.max(valid_values).detach().item(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        } if use_critic else {}),

        # response length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'response_length/mean':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(response_length).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'response_length/max':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.max(response_length).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'response_length/min':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.min(response_length).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'response_length/clip_ratio':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'prompt_length/mean':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(prompt_length).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'prompt_length/max':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.max(prompt_length).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'prompt_length/min':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.min(prompt_length).detach().item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'prompt_length/clip_ratio':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }

    # metrics for actions
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if 'turns_stats' in batch.meta_info:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if 'active_mask' in batch.meta_info:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if 'valid_action_stats' in batch.meta_info:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if 'valid_search_stats' in batch.meta_info:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return metrics


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_timing_metrics(batch, timing_raw):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_info = _compute_response_info(batch)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_response_tokens = torch.sum(response_info['response_length']).item()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_tokens_of_section = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'gen': num_response_tokens,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **{
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **{
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            f'timing_s/{name}': value for name, value in timing_raw.items()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **{
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )) & set(timing_raw.keys())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        },
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@contextmanager
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _timer(name: str, timing_raw: Dict[str, float]):
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with Timer(name=name, logger=None) as timer:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        yield
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    timing_raw[name] = timer.last


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RayPPOTrainer(object):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 config,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 tokenizer,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 role_worker_mapping: dict[Role, WorkerType],
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 resource_pool_manager: ResourcePoolManager,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 reward_fn=None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.reward_fn = reward_fn
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.val_reward_fn = val_reward_fn

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.hybrid_engine:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.role_worker_mapping = role_worker_mapping
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.resource_pool_manager = resource_pool_manager
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_rm = Role.RewardModel in role_worker_mapping
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_reference_policy:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if config.algorithm.kl_ctrl.type == 'fixed':
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            # 中文注释：下一行继续判断其他条件分支。
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise NotImplementedError
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._create_dataloader()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_logger()
    
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_logger(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.tracking import Tracking
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                          experiment_name=self.config.trainer.experiment_name,
                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                          default_backend=self.config.trainer.logger,
                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                          config=OmegaConf.to_container(self.config, resolve=True))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _create_dataloader(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         tokenizer=self.tokenizer,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         prompt_key=self.config.data.prompt_key,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         filter_prompts=True,
                                         # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         truncation='error')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.data.train_data_num is not None:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           batch_size=self.config.data.train_batch_size,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           drop_last=True,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           collate_fn=collate_fn)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       tokenizer=self.tokenizer,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       prompt_key=self.config.data.prompt_key,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       filter_prompts=True,
                                       # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       truncation='error')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.data.val_data_num is not None:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         batch_size=self.config.data.val_batch_size,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         shuffle=False,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         drop_last=True,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         collate_fn=collate_fn)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(self.train_dataloader) >= 1
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.trainer.total_training_steps is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_training_steps = self.config.trainer.total_training_steps

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.total_training_steps = total_training_steps
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Total training steps: {self.total_training_steps}')

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        OmegaConf.set_struct(self.config, True)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with open_dict(self.config):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.critic.optim.total_training_steps = total_training_steps

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _validate(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_tensor_lst = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_source_lst = []

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gen_config = GenerationConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_turns=self.config.max_turns,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_start_length=self.config.data.max_start_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_prompt_length=self.config.data.max_prompt_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_response_length=self.config.data.max_response_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_obs_length=self.config.data.max_obs_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            no_think_rl=self.config.algorithm.no_think_rl,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            search_url = self.config.retriever.url,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            topk = self.config.retriever.topk,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # Agent config preparation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        generation_manager = LLMGenerationManager(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer=self.tokenizer,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_rollout_wg=self.actor_rollout_wg,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            config=gen_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            is_validation = True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.config.do_search:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for test_data in self.val_dataloader:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return {}

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_gen_batch.meta_info = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'eos_token_id': self.tokenizer.eos_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'pad_token_id': self.tokenizer.pad_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'recompute_log_prob': False,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'do_sample': False,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'validate': True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }

                # pad to be divisible by dp_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print('validation generation end')

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                reward_tensor = self.val_reward_fn(test_batch)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                reward_tensor_lst.append(reward_tensor)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for batch_dict in self.val_dataloader:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                timing_raw = {}
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                test_gen_batch.meta_info = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'eos_token_id': self.tokenizer.eos_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'pad_token_id': self.tokenizer.pad_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'recompute_log_prob': False,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'do_sample': False,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'validate': True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }
                # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                with _timer('step', timing_raw):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                    with _timer('gen', timing_raw):
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        generation_manager.timing_raw = timing_raw
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            gen_batch=test_gen_batch,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            initial_input_ids=first_input_ids,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        )
                    
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                    for key in test_batch.batch.keys():
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    reward_tensor = self.val_reward_fn(test_batch)

                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    reward_tensor_lst.append(reward_tensor)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_source_reward = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(reward_tensor.shape[0]):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_source = data_sources[i]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if data_source not in data_source_reward:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_source_reward[data_source] = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data_source_reward[data_source].append(reward_tensor[i].item())

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metric_dict = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for data_source, rewards in data_source_reward.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return metric_dict


    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_workers(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Init resource pool and worker group"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.resource_pool_manager.create_resource_pool()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.hybrid_engine:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     config=self.config.actor_rollout_ref,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     role='actor_rollout')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError

        # create critic
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.algorithm.adv_estimator == 'gae':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.use_critic = True
            
        # 中文注释：下一行继续判断其他条件分支。
        elif self.config.algorithm.adv_estimator == 'grpo':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.use_critic = False
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError

        # create reference policy if needed
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_reference_policy:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  config=self.config.actor_rollout_ref,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  role='ref')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_rm:
            # we create a RM here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_wg = {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.wg_dicts = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.wg_dicts.append(wg_dict)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_critic:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_wg = all_wg['critic']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_wg.init_model()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_reference_policy:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ref_policy_wg = all_wg['ref']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ref_policy_wg.init_model()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_rm:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.rm_wg = all_wg['rm']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_rollout_wg = all_wg['actor_rollout']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_rollout_wg.init_model()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _save_checkpoint(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        f'global_step_{self.global_steps}')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.trainer.default_hdfs_dir, 'actor')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_critic:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             f'global_step_{self.global_steps}')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.config.trainer.default_hdfs_dir, 'critic')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = batch.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = attention_mask.shape[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = self.actor_rollout_wg.world_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                              k_partitions=world_size,
                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch.reorder(global_idx)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                    partitions=global_partition_lst,
                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                    prefix=logging_prefix)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics.update(global_balance_stats)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def fit(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger = self.logger
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            val_metrics = self._validate()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pprint(f'Initial validation metrics: {val_metrics}')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.log(data=val_metrics, step=self.global_steps)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.trainer.get('val_only', False):
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return

        # we start from step 1
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.global_steps += 1

        # Agent config preparation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gen_config = GenerationConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_turns=self.config.max_turns,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_start_length=self.config.data.max_start_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_prompt_length=self.config.data.max_prompt_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_response_length=self.config.data.max_response_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_obs_length=self.config.data.max_obs_length,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            no_think_rl=self.config.algorithm.no_think_rl,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            search_url = self.config.retriever.url,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            topk = self.config.retriever.topk,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        generation_manager = LLMGenerationManager(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer=self.tokenizer,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_rollout_wg=self.actor_rollout_wg,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            config=gen_config,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # start training loop
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for epoch in range(self.config.trainer.total_epochs):
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for batch_dict in self.train_dataloader:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'epoch {epoch}, step {self.global_steps}')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metrics = {}
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                timing_raw = {}

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                with _timer('step', timing_raw):
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if not self.config.do_search:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    # 中文注释：下一行处理前面条件都不满足时的默认分支。
                    else:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('gen', timing_raw):
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            generation_manager.timing_raw = timing_raw
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                gen_batch=gen_batch,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                initial_input_ids=first_input_ids,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                        for key in final_gen_batch_output.batch.keys():
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with torch.no_grad():
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                    for key in batch.batch.keys():
                        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                        if key != 'old_log_probs':
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch.batch[key] = batch.batch[key].long()

                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.use_reference_policy:
                        # compute reference log_prob
                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('ref', timing_raw):
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch = batch.union(ref_log_prob)

                    # compute values
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.use_critic:
                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('values', timing_raw):
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            values = self.critic_wg.compute_values(batch)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch = batch.union(values)

                    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                        if self.use_rm:
                            # we first compute reward model score
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        reward_tensor = self.reward_fn(batch)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            metrics.update(kl_metrics)
                        # 中文注释：下一行处理前面条件都不满足时的默认分支。
                        else:
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        batch = compute_advantage(batch,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  gamma=self.config.algorithm.gamma,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  lam=self.config.algorithm.lam,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.use_critic:
                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('update_critic', timing_raw):
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            critic_output = self.critic_wg.update_critic(batch)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('update_actor', timing_raw):
                            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        metrics.update(actor_output_metrics)

                    # validate
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('testing', timing_raw):
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            val_metrics: dict = self._validate()
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        metrics.update(val_metrics)

                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
                        with _timer('save_checkpoint', timing_raw):
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            self._save_checkpoint()

                # collect metrics
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logger.log(data=metrics, step=self.global_steps)

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.global_steps += 1

                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if self.val_reward_fn is not None:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        val_metrics = self._validate()
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        pprint(f'Final validation metrics: {val_metrics}')
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        logger.log(data=val_metrics, step=self.global_steps)
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return
    
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _create_loss_mask(self, batch, metrics):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Create loss mask for state tokens."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = batch.batch['responses'].shape[-1]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch.batch['loss_mask'] = loss_mask

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics.update({
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'state_tokens/total': loss_mask.sum().item(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        })
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return batch, metrics
