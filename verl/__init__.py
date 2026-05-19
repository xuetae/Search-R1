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
veRL - Volcano Engine Reinforcement Learning
==============================================
一个用于LLM强化学习训练的框架，支持：
- PPO、GRPO等多种RL算法
- 分布式训练（支持多GPU/多节点）
- 高效的张量操作和数据处理
- 与多种LLM模型兼容

核心概念：
- DataProto: 数据传输协议
- Worker: 执行单元（Actor、Critic等）
- Trainer: 训练协调器
"""

import os  # 文件路径操作

# 获取verl包的根目录
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# 读取版本号
with open(os.path.join(version_folder, 'version/version')) as f:
    __version__ = f.read().strip()

# 导入核心数据协议
from .protocol import DataProto

# 导入日志工具
from .utils.logging_utils import set_basic_config
import logging

# 配置基础日志级别
set_basic_config(level=logging.WARNING)
