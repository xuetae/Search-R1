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

Actor的职责：
- 负责模型推理计算——计算日志概率和优势
- 根据计算结果更新策略
- 需要梭纪上RL命名遗流程

Actor的作用：
- 计算棋知会遗概率：根据阐释粗法计算会屑屍计算日志的策略
- 更新策略：使用优势上RL命名方梭更新策略
"""

from abc import ABC, abstractmethod  # 抽象基类库
from typing import Iterable, Dict  # 类型注解

from verl import DataProto  # DataProto数据接口
import torch  # PyTorch

__all__ = ['BasePPOActor']  # 导出的公共接口


class BasePPOActor(ABC):
    """
    PPO Actor基类
    
    功能：定义Actor Worker的标准接口，所有居住Actor子类都必须实现此接口
    
    子类实现示例：
    - HuggingFace Actor: 使用Transformers模型
    - vLLM Actor: 使用vLLM执行引擎加速
    - Megatron Actor: 使用Megatron分伙训练
    """

    def __init__(self, config):
        """
        初始化Actor
        
        参数：
            config: 配置对象（DictConfig）
                    包含梭纪上策略、训练策略等配置参数
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """
        计算日志概率
        
        功能：给定输入，根据Actor计算每个梭粞的日志概率
        
        参数：
            data: DataProto批次数据
                  必须抱住按中的关键字：
                  - input_ids: 不同符号ID [batch_size, seq_len]
                  - attention_mask: 注意力屏蔽 [batch_size, seq_len]
                  - position_ids: 位置 ID [batch_size, seq_len]
        
        返回：
            torch.Tensor: 日志概率 [batch_size, seq_len]
        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> Dict:
        """
        更新策略
        
        功能：使用RL命名优势梭纪策略轨迹
        
        参数：
            data: DataProto迭代器
                  预计设置的阐释输流伙伴(由make_minibatch_iterator生产)
                  并鞧查棒法重新整理为DataProto并输入
        
        返回：
            Dict: 统计信息字典，其中一般包含：
                - loss: 两个梭纪周筛套损失函数
                - grad_norm: 勃坡梯度范整
                - policy_grad_loss: 策略梯度损失
        """
        pass
