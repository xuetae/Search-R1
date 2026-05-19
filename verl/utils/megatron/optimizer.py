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
from apex.optimizers import FusedAdam as Adam
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from apex.optimizers import FusedSGD as SGD
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.optimizer.distrib_optimizer import DistributedOptimizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.optimizer import get_param_groups

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.optimizer_config import OptimizerConfig


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_megatron_optimizer(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config: OptimizerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        no_weight_decay_cond=None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scale_lr_cond=None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lr_mult=1.0,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        check_for_nan_in_loss_and_grad=False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        overlap_param_gather=False  # add for verl
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
):
    # Base optimizer.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    param_groups = get_param_groups(model, no_weight_decay_cond, scale_lr_cond, lr_mult)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.optimizer == 'adam':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        optimizer = Adam(param_groups,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         lr=config.lr,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         weight_decay=config.weight_decay,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         betas=(config.adam_beta1, config.adam_beta2),
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         eps=config.adam_eps)
    # 中文注释：下一行继续判断其他条件分支。
    elif config.optimizer == 'sgd':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        optimizer = SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise Exception('{} optimizer is not supported.'.format(config.optimizer))

    # Determine whether the params have main-grad field.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    params_have_main_grad = True

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        grad_scaler = None

        # Constant loss scale.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if config.loss_scale:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if config.fp16:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                grad_scaler = DynamicGradScaler(initial_scale=config.initial_loss_scale,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                min_scale=config.min_loss_scale,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                growth_factor=2.0,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                backoff_factor=0.5,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                growth_interval=config.loss_scale_window,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                hysteresis=config.hysteresis)

        # Megatron optimizer.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if config.use_distributed_optimizer:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return DistributedOptimizer(optimizer, config.clip_grad, config.log_num_zeros_in_grad,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        check_for_nan_in_loss_and_grad, params_have_main_grad, config.fp16, config.bf16,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        config.params_dtype, grad_scaler, model, overlap_param_gather)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return Float16OptimizerWithFloat16Params(optimizer, config.clip_grad, config.log_num_zeros_in_grad,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     check_for_nan_in_loss_and_grad, params_have_main_grad, config.fp16,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     config.bf16, config.params_dtype, grad_scaler, model)

    # FP32.
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return FP32Optimizer(optimizer, config.clip_grad, config.log_num_zeros_in_grad, check_for_nan_in_loss_and_grad,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         params_have_main_grad, model)
