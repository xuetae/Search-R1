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

import os  # 操作系统接口（环境变量）


def initialize_global_process_group(timeout_second=36000):
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
    import torch.distributed  # PyTorch分布式模块
    from datetime import timedelta  # 超时设置
    
    # 初始化NCCL后端（GPU通信）
    torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    
    # 从环境变量获取rank信息
    local_rank = int(os.environ["LOCAL_RANK"])  # 本机GPU编号
    rank = int(os.environ["RANK"])  # 全局进程编号
    world_size = int(os.environ["WORLD_SIZE"])  # 总进程数

    # 设置进程使用的GPU
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)  # 将进程绑定到local_rank对应的GPU
    
    return local_rank, rank, world_size
