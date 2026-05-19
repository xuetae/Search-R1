# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Callable, Optional

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class OptimizerConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Configuration for optimizer."""

    ##############
    # General
    ##############
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    optimizer: str = 'adam'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Optimizer to use (one of Adam or SGD)."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lr: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_lr: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    decoupled_lr: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Separate learning rate for the input and output layer."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    decoupled_min_lr: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Minimum value for learning rate for the input and output layer. The scheduler clip values
       below this threshold.
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_decay: float = 0.01
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Weight decay coefficient for L2 regularization."""

    ##############
    # Precision
    ##############
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    fp16: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """If true, train with fp16 mixed precision training. Defaults to False."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bf16: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """If true, train with bf16 mixed precision training. Defaults to False."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_dtype: torch.dtype = torch.float32
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """dtype used when intializing the weights. Defaults to torch.float32."""

    ###############
    # Loss scaling
    ###############
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    loss_scale: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    initial_loss_scale: float = 2**32
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Initial loss-scale for dynamic loss scaling."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_loss_scale: float = 1.0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Minimum loss scale for dynamic loss scaling."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    loss_scale_window: float = 1000
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Window over which to raise/lower dynamic scale."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hysteresis: int = 2
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Hysteresis for dynamic loss scaling."""

    ##############
    # Optimizer
    ##############
    # Adam
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    adam_beta1: float = 0.9
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    adam_beta2: float = 0.999
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    adam_eps: float = 1e-08
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    # SGD.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sgd_momentum: float = 0.9
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Momentum factor for SGD optimizer."""

    #######################
    # Distributed optimizer
    #######################
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    use_distributed_optimizer: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Distribute optimizer state over data-parallel replicas."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    overlap_grad_reduce: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """If true, overlap grad reduce-scatter with backward compute in distributed optimizer."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    overlap_param_gather: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """If true, overlap param all-gather with forward compute in distributed optimizer."""

    ################
    # Miscellaneous
    ################
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    clip_grad: float = 1.0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Gradient clipping based on global L2 norm."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    log_num_zeros_in_grad: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """If true, calculate and log the number of zeros in gradient."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    barrier_with_L1_time: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """If true, use barrier with level 1 time measurements."""

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    timers: Callable = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Function to get timers."""
