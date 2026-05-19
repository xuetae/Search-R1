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
Critic Worker基类 (Critic Base Class)
=====================================
功能：定义Critic Worker的标准接口

Critic是强化学习训练中的关键组件，职责包括：
1. 估计价值函数：评估在给定状态下的长期收益
2. 计算优势信号：提供Actor学习的基线
3. 支持分布式训练：可在多GPU或多节点上运行

Critic模型通常是一个因果语言模型加上价值头（Value Head），用于估计每个令牌位置的价值。
Critic与Actor协同工作：Critic提供价值基线，Actor基于优势信号更新策略。
价值函数的准确性直接影响RL训练的稳定性和效率。
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from abc import ABC, abstractmethod  # 抽象基类库

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto  # DataProto数据接口

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['BasePPOCritic']  # 导出的公共接口


# 中文注释：下一行定义类，用于组织相关状态与行为。
class BasePPOCritic(ABC):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    PPO Critic基类
    
    功能：定义所有Critic实现必须遵循的标准接口
    
    子类实现示例：
    - HuggingFace Critic: 使用Transformers库模型加价值头
    - vLLM Critic: 使用vLLM推理引擎加速
    - Megatron Critic: 使用Megatron进行模型并行训练
    
    Critic的生命周期：
    1. 初始化：加载模型和分词器，添加价值头
    2. 推理：调用compute_values计算价值估计
    3. 反向传播：调用update_critic进行参数更新
    4. 重复2-3直到收敛
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化Critic
        
        参数：
            config: 配置对象（OmegaConf DictConfig）
                    包含模型路径、学习率等配置参数
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_values(self, data: DataProto) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        计算价值估计
        
        功能：给定输入序列，计算Critic对每个令牌位置的价值估计
        
        参数：
            data: DataProto批次数据，包含：
                  - input_ids: 输入令牌ID，形状 [batch_size, seq_len]
                  - attention_mask: 注意力掩码，形状 [batch_size, seq_len]
                  - position_ids: 位置ID，形状 [batch_size, seq_len]
        
        返回：
            torch.Tensor: 价值估计张量，形状 [batch_size, seq_len]
                          表示每个位置的价值函数估计值
                          
        说明：
        - 价值函数V(s)估计当前状态的长期累积奖励
        - 用于计算广义优势估计（GAE）
        - 需要忽略pad token的贡献（通过attention_mask过滤）
        - 返回的价值用于计算：advantage = reward + gamma * next_value - value
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @abstractmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_critic(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        更新Critic网络
        
        功能：使用实际收益（回报）对价值函数进行梯度更新
        
        参数：
            data: DataProto数据迭代器
                  包含以下信息用于Critic更新：
                  - input_ids: 输入令牌
                  - values: 旧Critic的价值估计
                  - returns: 计算出的累积回报（目标值）
                  - attention_mask: 掩码（标记有效令牌）
        
        返回：
            Dict: 训练统计信息字典，一般包含：
                  - critic_loss: 价值函数损失
                  - grad_norm: 梯度范数
                  - value_clipfrac: 被裁剪的比例
                  
        说明：
        - 实现Critic损失函数（MSE或Huber损失）：
          L_critic = MSE(V(s), returns)
        - 通常使用价值函数裁剪（Value Clipping）
        - 返回的统计信息用于监控Critic训练进度
        - Critic的好坏直接影响Actor的优势计算准确性
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass
