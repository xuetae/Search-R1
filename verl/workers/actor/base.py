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

"""
Actor Worker基类 (Actor Base Class)
=================================
功能：定义Actor Worker的标准接口

Actor是强化学习训练中的关键组件，职责包括：
1. 执行模型前向推理：计算给定输入的日志概率
2. 计算策略梯度：基于优势信号更新策略
3. 支持分布式训练：可在多GPU或多节点上运行

Actor模型通常是一个因果语言模型（Causal LM），在RL训练中被更新以优化任务奖励。
Actor与Critic（价值函数）协同工作：Actor生成动作，Critic估计价值基线。
"""

from abc import ABC, abstractmethod  # 抽象基类库
from typing import Iterable, Dict  # 类型注解

from verl import DataProto  # DataProto数据接口
import torch  # PyTorch

__all__ = ['BasePPOActor']  # 导出的公共接口


class BasePPOActor(ABC):
    """
    PPO Actor基类
    
    功能：定义所有Actor实现必须遵循的标准接口
    
    子类实现示例：
    - HuggingFace Actor: 使用Transformers库的模型
    - vLLM Actor: 使用vLLM推理引擎加速
    - Megatron Actor: 使用Megatron进行模型并行训练
    
    Actor的生命周期：
    1. 初始化：加载模型和分词器
    2. 推理：调用compute_log_prob计算概率
    3. 更新：调用update_policy进行梯度步
    4. 重复2-3直到收敛
    """

    def __init__(self, config):
        """
        初始化Actor
        
        参数：
            config: 配置对象（OmegaConf DictConfig）
                    包含模型路径、学习率等配置参数
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """
        计算日志概率
        
        功能：给定输入序列，计算当前策略下每个令牌的日志概率
        
        参数：
            data: DataProto批次数据，包含：
                  - input_ids: 输入令牌ID，形状 [batch_size, seq_len]
                  - attention_mask: 注意力掩码，形状 [batch_size, seq_len]
                  - position_ids: 位置ID，形状 [batch_size, seq_len]
        
        返回：
            torch.Tensor: 日志概率张量，形状 [batch_size, seq_len]
                          表示每个位置每个令牌的日志概率值
                          
        说明：
        - 用于计算PPO中的log_prob（当前策略的对数概率）
        - 通常通过softmax后取log得到
        - 需要忽略pad token的贡献（通过attention_mask过滤）
        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        """
        更新策略网络
        
        功能：使用RL优势信号对策略进行梯度更新
        
        参数：
            data: DataProto数据迭代器
                  包含以下信息用于策略更新：
                  - input_ids: 输入令牌
                  - old_log_prob: 旧策略的日志概率（用于PPO比率计算）
                  - advantages: 优势估计（通常来自Critic）
                  - attention_mask: 掩码（标记有效令牌）
        
        返回：
            Dict: 训练统计信息字典，一般包含：
                  - loss: 策略梯度损失
                  - grad_norm: 梯度范数
                  - policy_loss: PPO损失
                  - kl_div: KL散度指标
                  
        说明：
        - 实现PPO损失函数：
          L_policy = -E[min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)]
        - 其中 ratio = exp(log_prob - old_log_prob)
        - 通常需要配合熵正则化和KL惩罚
        - 返回的统计信息用于监控训练进度
        """
        pass
