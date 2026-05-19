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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 文件路径操作

# 获取verl包的根目录
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# 读取版本号
# 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
with open(os.path.join(version_folder, 'version/version')) as f:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    __version__ = f.read().strip()

# 导入核心数据协议
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .protocol import DataProto

# 导入日志工具
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .utils.logging_utils import set_basic_config
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging

# 配置基础日志级别
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
set_basic_config(level=logging.WARNING)
