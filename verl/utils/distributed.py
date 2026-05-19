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
分布式训练工具 (Distributed Training Utils)
=========================================
功能：初始化分布式训练环境

主要功能：
- 初始化PyTorch分布式进程组
- 获取进程等级和世界大小
- 设置GPU设备映射

依赖环境变量：
- LOCAL_RANK: 本地GPU序号
- RANK: 全局进程序号
- WORLD_SIZE: 总进程数
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 操作系统接口（环境变量）


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def initialize_global_process_group(timeout_second=36000):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    初始化分布式进程组
    
    功能：设置PyTorch分布式计算环境，通常在多卡或多机训练时调用
    
    参数：
        timeout_second: 进程组通信超时时间（秒），默认10小时
        
    返回：
        (local_rank, rank, world_size) 元组
        - local_rank: 本机内的GPU编号（0, 1, ...）
        - rank: 全局进程编号（0到world_size-1）
        - world_size: 总进程数（总GPU数）
        
    工作流程：
    1. 初始化NCCL后端进程组
    2. 从环境变量获取rank信息
    3. 将进程与对应GPU绑定
    
    说明：
    - 通常由torchrun或torch.distributed.launch启动
    - 这些工具自动设置环境变量
    - NCCL: NVIDIA Collective Communications Library
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import torch.distributed  # PyTorch分布式模块
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from datetime import timedelta  # 超时设置
    
    # 初始化NCCL后端（GPU通信）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    
    # 从环境变量获取rank信息
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_rank = int(os.environ["LOCAL_RANK"])  # 本机GPU编号
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank = int(os.environ["RANK"])  # 全局进程编号
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    world_size = int(os.environ["WORLD_SIZE"])  # 总进程数

    # 设置进程使用的GPU
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.is_initialized():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.set_device(local_rank)  # 将进程绑定到local_rank对应的GPU
    
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return local_rank, rank, world_size
