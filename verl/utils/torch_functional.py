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
Contain small torch utilities
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Union, List, Optional

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn.functional as F
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn

# 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
try:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
# 中文注释：下一行处理异常分支，保证错误可控。
except ImportError:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def gather_from_labels(data, label):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def logprobs_from_logits(logits, labels):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_dim = logits.shape[:-1]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        last_dim = logits.shape[-1]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.reshape(-1, last_dim)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        labels = labels.reshape(-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = logprobs_from_logits_flash_attn(logits, labels)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.view(*batch_dim)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = logprobs_from_logits_naive(logits, labels)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def logprobs_from_logits_flash_attn(logits, labels):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = -cross_entropy_loss(logits, labels)[0]
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def logprobs_from_logits_naive(logits, labels):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logp = F.log_softmax(logits, dim=-1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logpy = gather_from_labels(logp, labels)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return logpy


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def logprobs_of_labels_v2(logits: torch.FloatTensor, labels):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    A memory efficient implementation of logprobs_from_logits
    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert logits.dtype == torch.float32, 'Using bf16 logits with logprobs_of_labels_v2 may lead to divergence'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logprobs_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logprobs_labels = logprobs_labels - torch.logsumexp(logits, dim=-1, keepdim=True)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return logprobs_labels.squeeze(-1)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def clip_by_value(x, tensor_min, tensor_max):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return clipped


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def entropy_from_logits(logits: torch.Tensor):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Calculate entropy from logits."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pd = torch.nn.functional.softmax(logits, dim=-1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return entropy


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def masked_sum(values, mask, axis=None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute mean of tensor with a masked values."""
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return (values * mask).sum(axis=axis)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def masked_mean(values, mask, axis=None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute mean of tensor with a masked values."""
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def masked_var(values, mask, unbiased=True):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute variance of tensor with masked values."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mean = masked_mean(values, mask)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    centered_values = values - mean
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    variance = masked_mean(centered_values**2, mask)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if unbiased:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mask_sum = mask.sum()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if mask_sum == 0:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if mask_sum == 1:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        bessel_correction = mask_sum / (mask_sum - 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        variance = variance * bessel_correction
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return variance


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def masked_whiten(values, mask, shift_mean=True):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Whiten values with masked values."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not shift_mean:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        whitened += mean
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return whitened


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    '''
    e.g. end of sentence token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    eos_mask = response_id.eq(eos_token).long()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return eos_mask


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_grad_norm(model: nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_grad_square = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_params = 0
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for param in model.parameters():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if param.grad is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return total_grad_square


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def broadcast_dict_tensor(tensors: Union[Dict[str, torch.Tensor], TensorDict], src, group):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    TODO: optimize this. Technically, we only need one broadcast
    """

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key in tensors.sorted_keys:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def allgather_dict_tensors(tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    TODO: optimize this.
    - We can use async ops
    - We can use only one allgather
    Args:
        tensors:
        size:
        group:

    Returns:

    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(tensors, TensorDict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_tensor_dict = True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors_as_dict = tensors.to_dict()
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors_as_dict = tensors
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_tensor_dict = False

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sorted_keys = sorted(tensors_as_dict.keys())
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key in sorted_keys:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        val = tensors_as_dict[key]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output[key] = [torch.empty_like(val) for _ in range(size)]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output[key] = torch.cat(output[key], dim=dim)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if is_tensor_dict:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert tensors.batch_size[0] % batch_size == 0, \
        f'input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}'
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return tensors.split(batch_size)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tensors.shape[-1] >= max_seq_len:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return tensors
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PreTrainedTokenizer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def tokenize_and_postprocess_data(prompt: str,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  tokenizer: PreTrainedTokenizer,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  max_length: int,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  pad_token_id: int,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  left_pad=True,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  truncation='error'):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    input_data is the output from tokenizer.
    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert truncation in ['left', 'right', 'error']

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids = input_data['input_ids']
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attention_mask = input_data['attention_mask']

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert input_ids.ndim == 2

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sequence_length = input_ids.shape[-1]
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if sequence_length < max_length:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = pad_sequence_to_length(input_ids,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           max_seq_len=max_length,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           pad_token_id=pad_token_id,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           left_pad=left_pad)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = pad_sequence_to_length(attention_mask,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                max_seq_len=max_length,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                pad_token_id=0,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                left_pad=left_pad)
    # 中文注释：下一行继续判断其他条件分支。
    elif sequence_length > max_length:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = input_ids[:, -max_length:]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = attention_mask[:, -max_length:]
        # 中文注释：下一行继续判断其他条件分支。
        elif truncation == 'right':
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = input_ids[:, :max_length]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = attention_mask[:, :max_length]
        # 中文注释：下一行继续判断其他条件分支。
        elif truncation == 'error':
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return input_ids, attention_mask


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """ Remove the pad token. 

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    no_padding_batch = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for ids, mask in zip(input_ids, attention_mask):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        no_padding_batch.append((ids[len(ids) - mask.sum():]).cpu().numpy().tolist())
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return no_padding_batch


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def log_probs_from_logits_response(input_ids, logits, response_length):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute the response log_probs from full logits. Note that logits = model(input_ids)
    
    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]
    
    Returns:
        response_log_prob: 
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_logits = logits[:, -response_length - 1:-1]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response = input_ids[:, -response_length:]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return response_log_prob


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute the log_probs from logits with rmpad logits and pad input. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size
    
    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
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
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
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
def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Compute the log_probs from logits with rmpad input_ids and logits. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size
    
    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from flash_attn.bert_padding import pad_input
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # transpose back to [total_nnz, 1]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
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


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if temperature != 1.:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits.div_(temperature)  # inplace operation to avoid OOM
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return logits


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Optimizer related
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.optim import Optimizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.optim.lr_scheduler import LambdaLR
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import math


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_cosine_schedule_with_warmup(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    optimizer: Optimizer,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_warmup_steps: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_training_steps: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_lr_ratio: float = 0.0,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_cycles: float = 0.5,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    last_epoch: int = -1,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    coef = (1 - min_lr_ratio) * 0.5
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    intercept = (1 + min_lr_ratio) * 0.5

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def lr_lambda(current_step):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if current_step < num_warmup_steps:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return float(current_step) / float(max(1, num_warmup_steps))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return max(0.0, x * coef + intercept)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_constant_schedule_with_warmup(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    optimizer: Optimizer,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_warmup_steps: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    last_epoch: int = -1,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def lr_lambda(current_step):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return min(1, float(current_step) / float(max(1, num_warmup_steps)))

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    combined_attention_mask = None
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if input_shape[-1] > 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        combined_attention_mask = _make_causal_mask(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_shape,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            inputs_embeds.dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device=inputs_embeds.device,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          tgt_len=input_shape[-1]).to(inputs_embeds.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   combined_attention_mask)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Make causal mask used for bi-directional self-attention.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bsz, tgt_len = input_ids_shape
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask_cond = torch.arange(mask.size(-1), device=device)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    mask = mask.to(dtype)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    bsz, src_len = mask.size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tgt_len = tgt_len if tgt_len is not None else src_len

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    inverted_mask = 1.0 - expanded_mask

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_unpad_data(attention_mask):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return (
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        indices,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cu_seqlens,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_seqlen_in_batch,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
