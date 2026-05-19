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
"""
Utilities for using tensor_parallel in megatron
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.nn import init
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu, tensor_parallel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.torch_functional as verl_F


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_kwargs_with_config(dictionary: Dict, config: ModelParallelConfig):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dictionary['config'] = config
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return dictionary


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_default_kwargs_for_model_parallel_config():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_parallel_config_kwargs = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'params_dtype': torch.float32,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'use_cpu_initialization': False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'perform_initialization': True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'gradient_accumulation_fusion': False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'sequence_parallel': False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return model_parallel_config_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_default_model_parallel_config():
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_common_default_kwargs_for_parallel_linear():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    default_model_parallel_config = get_default_model_parallel_config()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    common_default_kwargs = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'init_method': init.xavier_normal_,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'stride': 1,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'keep_master_weight_for_test': False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'config': default_model_parallel_config,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return common_default_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_default_kwargs_for_column_parallel_linear():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    column_parallel_config_kwargs = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'async_tensor_model_parallel_allreduce': False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    column_default_kwargs = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'config': ModelParallelConfig(**model_parallel_config_kwargs),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    common_default_kwargs.update(column_default_kwargs)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return common_default_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_default_kwargs_for_row_parallel_linear():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return common_default_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_default_kwargs_for_parallel_embedding():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    embedding_default_kwargs = {
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'init_method': init.xavier_normal_,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        'config': ModelParallelConfig(**model_parallel_config_kwargs),
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    }
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return embedding_default_kwargs


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def is_tensor_parallel_param(param):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_parallel_partition_dim(param):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert is_tensor_parallel_param(param)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return param.partition_dim


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_tensor_parallel_partition_stride(param):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert is_tensor_parallel_param(param)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return param.partition_stride


# 中文注释：下一行定义类，用于组织相关状态与行为。
class _VocabParallelEntropy(torch.autograd.Function):

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        normalized_exp_logits = normalized_vocab_parallel_logits.exp()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        softmax_logits = normalized_exp_logits / normalized_sum_exp_logits
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return entropy.squeeze(dim=-1)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        grad_input = grad_output.unsqueeze(dim=-1) * softmax_logits * (sum_softmax_times_logits - vocab_parallel_logits)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return grad_input


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute entropy when the logits are sharded in tp ranks
    
    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)
        
    """
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def vocab_parallel_log_probs_from_logits(logits, labels):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from flash_attn.bert_padding import pad_input, unpad_input

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size, seqlen = input_ids.shape
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(logits=logits_rmpad,
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                labels=input_ids_rmpad_rolled)  # (total_nnz,)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            indices=indices,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            batch=batch_size,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            seqlen=seqlen)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def vocab_parallel_compute_entropy_loss(logits, eos_mask):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    entropy = vocab_parallel_entropy(logits)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return entropy_loss
