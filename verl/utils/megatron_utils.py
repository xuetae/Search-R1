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
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""Pretrain utilities."""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Any, Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import DictConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_dtypes import PrecisionType
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.memory_buffer import build_memory_reference_from_module
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn.functional as F

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import mpu, tensor_parallel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.utils import get_model_config
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.transformer import TransformerConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.transformer.module import Float16Module
# from megatron.core.distributed import DistributedDataParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.distributed import DistributedDataParallel as DDP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.enums import ModelType


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Build the model."""
    # Build model.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pre_process = mpu.is_pipeline_first_stage()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            post_process = mpu.is_pipeline_last_stage()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            this_model.model_type = model_type
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model.append(this_model)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pre_process = mpu.is_pipeline_first_stage()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        post_process = mpu.is_pipeline_last_stage()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        add_encoder = True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        add_decoder = True
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_type == ModelType.encoder_and_decoder:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rank = mpu.get_pipeline_model_parallel_rank()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                world_size = mpu.get_pipeline_model_parallel_world_size()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pre_process = rank == 0 or rank == split_rank
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                add_encoder = mpu.is_pipeline_stage_before_split()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                add_decoder = mpu.is_pipeline_stage_after_split()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model = model_provider_func(pre_process=pre_process,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        post_process=post_process,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        add_encoder=add_encoder,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        add_decoder=add_decoder)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model.model_type = model_type

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not isinstance(model, list):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for model_module in model:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for param in model_module.parameters():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if mpu.get_data_parallel_rank() == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(' > number of parameters on (tensor, pipeline) '
              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
              'model parallel rank ({}, {}): {}'.format(
                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                  mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(),
                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                  sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])),
              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
              flush=True)

    # GPU allocation.
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for model_module in model:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config = get_model_config(model[0])
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = [Float16Module(config, model_module) for model_module in model]

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if wrap_with_ddp:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = [
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            DDP(config=config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                module=model_chunk,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                accumulate_allreduce_grads_in_fp32=True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                overlap_grad_reduce=False,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_distributed_optimizer=True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                disable_bucketing=(model_chunk_idx > 0)) for (model_chunk_idx, model_chunk) in enumerate(model)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # if args.data_parallel_random_init:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for model_module in model:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_module.broadcast_params()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return model


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return_list = True
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not isinstance(model, list):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = [model]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return_list = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    unwrapped_model = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for model_module in model:
        # 中文注释：下一行开始循环，直到条件不再满足。
        while isinstance(model_module, module_instances):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_module = model_module.module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        unwrapped_model.append(model_module)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not return_list:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return unwrapped_model[0]
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return unwrapped_model


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def convert_config(hf_config: PretrainedConfig, megatron_config) -> TransformerConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f'megatron config {megatron_config}')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dt = PrecisionType.to_dtype(megatron_config['param_dtype'])
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f'pipeline_dtype=megatron_config {dt}')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    transformer_config = TransformerConfig(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_layers=hf_config.num_hidden_layers,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_size=hf_config.hidden_size,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_attention_heads=hf_config.num_attention_heads,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_query_groups=hf_config.num_key_value_heads,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ffn_hidden_size=hf_config.intermediate_size,
        #    max_position_embeddings=hf_config.max_position_embeddings,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        activation_func=F.silu,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        normalization='RMSNorm',
        #    rotary_percent=False, # default,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gated_linear_unit=True,  # for llama
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_cpu_initialization=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        apply_residual_connection_post_layernorm=False,  # check what's this mean
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        add_bias_linear=False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pipeline_dtype=PrecisionType.to_dtype(megatron_config['param_dtype']),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params_dtype=PrecisionType.to_dtype(megatron_config['param_dtype']),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sequence_parallel=megatron_config['sequence_parallel_enabled'],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        variable_seq_lengths=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        masked_softmax_fusion=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        bf16=PrecisionType.to_dtype(megatron_config['param_dtype']) is torch.bfloat16)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.get_rank() == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'tensor_parallel_size={transformer_config.tensor_model_parallel_size} \n \
                pipeline_model_parallel_size={transformer_config.pipeline_model_parallel_size} \n \
                virtual_pipeline_model_parallel_size={transformer_config.virtual_pipeline_model_parallel_size} \n \
                pipeline_dtype={transformer_config.pipeline_dtype} \n \
                params_dtype={transformer_config.params_dtype} \n \
                sequence_parallel={transformer_config.sequence_parallel} \n \
                variable_seq_lengths={transformer_config.variable_seq_lengths} \n \
                masked_softmax_fusion={transformer_config.masked_softmax_fusion} \n ')

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return transformer_config


# from megatron.core.optimizer import OptimizerConfig

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.optimizer_config import OptimizerConfig


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def init_megatron_optim_config(optim_config: Dict) -> OptimizerConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config = OptimizerConfig(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        optimizer='adam',
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lr=optim_config.get('lr'),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        clip_grad=optim_config.get('clip_grad'),
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_decay=1e-2,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        bf16=True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params_dtype=torch.bfloat16,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_distributed_optimizer=True,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return config


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def init_model_parallel_config(config: DictConfig) -> ModelParallelConfig:
    # TODO(sgm): check how to disable megatron timers
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    timers = FakeTimers()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return ModelParallelConfig(tensor_model_parallel_size=config.get('tensor_model_parallel_size'),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               pipeline_model_parallel_size=config.get('pipeline_model_parallel_size'),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               virtual_pipeline_model_parallel_size=config.get('virtual_pipeline_model_parallel_size'),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               sequence_parallel=config.get('sequence_parallel'),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               params_dtype=PrecisionType.to_dtype(config.get('param_dtype')),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               pipeline_dtype=PrecisionType.to_dtype(config.get('param_dtype')),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               bf16=True,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               fp16=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               timers=timers)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class FakeTimers:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Disable All Megatron Timing with FakeTimers"""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.timers import DummyTimer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dummy_timer = DummyTimer()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.dummy_timer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def offload_megatron_param_and_grad(module_list: nn.ModuleList, offload_grad=False, hybrid_engine=None):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if hybrid_engine is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for buffer in hybrid_engine.memory_buffers[pp_rank].values():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer.data = buffer.data.to('cpu', non_blocking=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        build_memory_reference_from_module(module_list, hybrid_engine.memory_buffers[pp_rank], maintain_weight=True)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for module in module_list:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for _, param in module.named_parameters():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param.data = param.data.to('cpu', non_blocking=True)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if offload_grad and param.grad is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    param.grad = param.grad.to("cpu", non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_megatron_param_and_grad(module_list: nn.ModuleList, device_id, load_grad=False, hybrid_engine=None):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if hybrid_engine is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for buffer in hybrid_engine.memory_buffers[pp_rank].values():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer.data = buffer.data.to(device_id, non_blocking=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        build_memory_reference_from_module(module_list, hybrid_engine.memory_buffers[pp_rank], maintain_weight=True)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for module in module_list:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for _, param in module.named_parameters():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                param.data = param.data.to(device_id, non_blocking=True)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if load_grad and param.grad is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    param.grad = param.grad.to(device_id, non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()
