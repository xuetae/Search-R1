# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
PPO算法核心函数 (PPO Core Algorithms)
================================
功能：实现PPO强化学习算法的核心计算函数

主要特性：
1. KL散度控制（自适应或固定）
2. GAE优势计算
3. PPO裁剪代理目标优化
4. GRPO结果监督优化算法

算法流程：
1. 收集体验数据（Rollout）
2. 计算广义优势估计（GAE）
3. 计算PPO损失函数
4. 进行梯度更新

参考资料：
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- AdaptiveKL: https://arxiv.org/pdf/1909.08593.pdf
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy数组
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from collections import defaultdict  # 默认字典

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.torch_functional as verl_F  # veRL的函数处理


# 中文注释：下一行定义类，用于组织相关状态与行为。
class AdaptiveKLController:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    自适应KL控制器
    参考论文：https://arxiv.org/pdf/1909.08593.pdf
    
    功能：根据实际KL散度与目标KL散度的差距来动态调整KL惩罚系数
    
    工作原理：
    1. 计算比例误差：(current_kl / target_kl - 1)
    2. 将误差限制在[-0.2, 0.2]范围内
    3. 计算乘数：1 + 比例误差 * n_steps / horizon
    4. 更新KL系数：value *= 乘数
    
    作用：确保训练过程中KL散度不会偏离目标值太远
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, init_kl_coef, target_kl, horizon):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化自适应KL控制器
        
        参数：
            init_kl_coef: 初始的KL惩罚系数（通常是较小的正数）
            target_kl: 目标KL散度值（通常0.01-0.05）
            horizon: 调度的时间步长（用于平滑更新）
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.value = init_kl_coef  # 当前的KL惩罚系数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.target = target_kl  # 目标KL散度
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.horizon = horizon  # 更新时间跨度

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update(self, current_kl, n_steps):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        更新KL惩罚系数
        
        参数：
            current_kl: 当前测量的KL散度值
            n_steps: 已执行的训练步数
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        target = self.target
        # 计算比例误差：相对于目标KL散度的偏差
        # 限制在[-0.2, 0.2]范围内，避免过度调整
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # 计算乘数：根据时间步长和误差大小调整KL系数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mult = 1 + proportional_error * n_steps / self.horizon
        # 更新KL系数：乘以计算出的乘数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.value *= mult  # 乘以乘法币更新KL系数


# 中文注释：下一行定义类，用于组织相关状态与行为。
class FixedKLController:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Fixed KL controller."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, kl_coef):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.value = kl_coef

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update(self, current_kl, n_steps):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_kl_controller(config): # seems never used?
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.critic.kl_ctrl.type == 'fixed':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    # 中文注释：下一行继续判断其他条件分支。
    elif config.critic.kl_ctrl.type == 'adaptive':
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       horizon=config.critic.kl_ctrl.horizon)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError('Unknown kl_ctrl type')

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return kl_ctrl


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 gamma: torch.Tensor, lam: torch.Tensor):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with torch.no_grad():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lastgaelam = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        advantages_reversed = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gen_len = token_level_rewards.shape[-1]

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for t in reversed(range(gen_len)):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lastgaelam = delta + gamma * lam * lastgaelam
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            advantages_reversed.append(lastgaelam)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        returns = advantages + values
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   eos_mask: torch.Tensor,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   index: torch.Tensor,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   epsilon: float = 1e-6):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_length = token_level_rewards.shape[-1]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_zero_mask = (token_level_rewards != 0)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    id2score = defaultdict(list)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    id2mean = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    id2std = {}

    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with torch.no_grad():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        bsz = scores.shape[0]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(bsz):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            id2score[index[i]].append(scores[i])
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx in id2score:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if len(id2score[idx]) == 1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                id2mean[idx] = torch.tensor(0.0)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                id2std[idx] = torch.tensor(1.0)
            # 中文注释：下一行继续判断其他条件分支。
            elif len(id2score[idx]) > 1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f"no score in prompt index: {idx}")
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(bsz):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return scores, scores


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    kl = old_log_prob - ref_log_prob
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return token_level_scores - kl * kl_ratio


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    negative_approx_kl = log_prob - old_log_prob
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ratio = torch.exp(negative_approx_kl)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pg_losses = -advantages * ratio
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return pg_loss, pg_clipfrac, ppo_kl


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_entropy_loss(logits, eos_mask):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return entropy_loss


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vf_losses1 = (vpreds - returns)**2
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vf_losses2 = (vpredclipped - returns)**2
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return vf_loss, vf_clipfrac


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if kl_penalty == "kl":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return logprob - ref_logprob

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if kl_penalty == "abs":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (logprob - ref_logprob).abs()

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if kl_penalty == "mse":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if kl_penalty == 'low_var_kl':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kl = ref_logprob - logprob
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ratio = torch.exp(kl)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kld = (ratio - kl - 1).contiguous()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return torch.clamp(kld, min=-10, max=10)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError

    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise NotImplementedError
