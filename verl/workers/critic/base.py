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
Critic Worker基类 (Critic Base Class)
=====================================
功能：定义Critic Worker的标准接口

Critic的职责：
- 负责估计无偏策略值（Value Function）
- 估计感受分技，作为优势上界（Baseline）
- 根据Target Value更新值函数

Critic的作用：
- 粗法估计：策略估计无名二值函数（优势上界）
- 更新估计器：根据实断收益计算损失函数并更新估计器
"""

from abc import ABC, abstractmethod  # 抽象基类库

import torch  # PyTorch

from verl import DataProto  # DataProto数据接口

__all__ = ['BasePPOCritic']  # 导出的公共接口


class BasePPOCritic(ABC):
    """
    PPO Critic基类
    
    功能：定义Critic Worker的标准接口，所有子类都必须实现此接口
    
    子类实现示例：
    - HuggingFace Critic: 使用Transformers模型（加上值头）
    - vLLM Critic: 使用vLLM执行引擎加速
    - Megatron Critic: 使用Megatron分伙训练
    """

    def __init__(self, config):
        """
        初始化Critic
        
        参数：
            config: 配置对象（DictConfig）
                    包含估计器配置参数
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_values(self, data: DataProto) -> torch.Tensor:
        """
        计算估计值
        
        功能：给定输入，计算每个梭粞的估计值（Value）
        
        参数：
            data: DataProto批次数据
                  包含无名符号、注意力屏蔽等
        
        返回：
            torch.Tensor: 估计值 [batch_size, seq_len]
                          每个梭粞的估计值
        """
        pass

    @abstractmethod
    def update_critic(self, data: DataProto):
        """
        更新估计器
        
        功能：根据实际收益（回报）计算粗法损失函数并更新估计器梭纪
        
        参数：
            data: DataProto迭代器
                  呃包含data吗（输入、收益、粗法值等）
        
        返回：
            Dict: 更新统计信息（可选），一般包括：
                - critic_loss: 估计器损失函数
                - grad_norm: 梯度范整
        """
        pass
