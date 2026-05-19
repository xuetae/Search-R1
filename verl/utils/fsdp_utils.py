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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import functools
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import math
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import itertools
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from contextlib import contextmanager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers.trainer_pt_utils import get_module_class_from_name
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def init_fn(x: torch.nn.Module):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not torch.distributed.get_rank() == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_init_weight_context_manager(use_meta_tensor=True):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from accelerate import init_empty_weights
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cpu_init_weights = lambda: torch.device('cpu')
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if use_meta_tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_context = cpu_init_weights
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Get FSDP wrap policy for the module.
    
    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config = {}

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.get('disable', False):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                    default_transformer_cls_names_to_wrap)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_num_params = config.get('min_num_params', 0)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    auto_wrap_policy = None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    policies = []

    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if is_lora:

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def lambda_policy_fn(module):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    module.weight.requires_grad):
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return True
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return False

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        policies.append(lambda_policy)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if min_num_params > 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        policies.append(size_policy)
    # 中文注释：下一行继续判断其他条件分支。
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformer_cls_to_wrap = set()
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformer_cls = get_module_class_from_name(module, layer_class)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if transformer_cls is None:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                transformer_cls_to_wrap.add(transformer_cls)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        transformer_policy = functools.partial(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformer_auto_wrap_policy,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            transformer_layer_cls=transformer_cls_to_wrap,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        policies.append(transformer_policy)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(policies) > 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return auto_wrap_policy


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def offload_fsdp_grad(module):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for _, param in module.named_parameters():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if param.grad is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param.grad = param.grad.to("cpu", non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_fsdp_grad(module, device_id):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for _, param in module.named_parameters():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if param.grad is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param.grad = param.grad.to(device_id, non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def offload_fsdp_param_and_grad(module, offload_grad=False):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for _, param in module.named_parameters():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hasattr(param, "_local_shard"):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param._local_shard = param._local_shard.to("cpu", non_blocking=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param.data = param.data.to('cpu', non_blocking=True)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if offload_grad and param.grad is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param.grad = param.grad.to("cpu", non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_fsdp_param_and_grad(module, device_id, load_grad=False):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for _, param in module.named_parameters():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hasattr(param, "_local_shard"):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param._local_shard = param._local_shard.to(device_id, non_blocking=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param.data = param.data.to(device_id, non_blocking=True)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_grad and param.grad is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param.grad = param.grad.to(device_id, non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def offload_fsdp_optimizer(optimizer):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for param_group in optimizer.param_groups:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for param in param_group['params']:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state = optimizer.state[param]
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for key, value in state.items():
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if isinstance(value, torch.Tensor):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    state[key] = value.to("cpu", non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_fsdp_optimizer(optimizer, device_id):
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for param_group in optimizer.param_groups:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for param in param_group['params']:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state = optimizer.state[param]
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for key, value in state.items():
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if isinstance(value, torch.Tensor):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    state[key] = value.to(device_id, non_blocking=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.cuda.empty_cache()


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@contextmanager
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def meta_device_init():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Create model parameters with meta device.

    Note buffers in model will still be initialized in default device (e.g., CPU),
    since the buffers can be non-persistent and filled with expected values that can
    NOT be captured in meta device.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device = torch.device("meta")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    old_register_parameter = nn.Module.register_parameter
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    registered = set()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def register_empty_parameter(module, name, param):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        old_register_parameter(module, name, param)
        # we will skip register shared parameters as it
        # is already registered previously
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if param is not None and param not in registered:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_cls = type(module._parameters[name])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kwargs = module._parameters[name].__dict__
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kwargs["requires_grad"] = param.requires_grad
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            registered.add(module._parameters[name])

    # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
    try:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nn.Module.register_parameter = register_empty_parameter
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        yield
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    finally:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        registered.clear()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nn.Module.register_parameter = old_register_parameter


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def parallel_load_safetensors(filepath):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Parallel load safetensors from huggingface checkpoint

    Huggingface checkpoint contains:

    - config.json: a json file for model configuration
    - model.safetensor.index.json: a json file for safetensors (parameters & buffers) index
    - model-000x-of-ooxx.safetensors: a binary file for safetensors (parameters & buffers) chunks

    Or (when model is small),

    - model.safetensors: a binary file for all parameters and buffers

    Each rank will own a part of model chunks and load them directly into GPU memory.
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from safetensors.torch import load_file

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    safetensors2param = {}

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    index_file = os.path.join(filepath, "model.safetensors.index.json")
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if os.path.exists(index_file):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        index = json.load(open(index_file, "rb"))
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for param_name, filename in index["weight_map"].items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            safetensors2param.setdefault(filename, []).append(param_name)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # in this case, the model is small and we can load it all at once
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param_file = os.path.join(filepath, "model.safetensors")
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        states = load_file(param_file)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for param_name in states:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        del states

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_files = len(safetensors2param)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ckpt_chunks = sorted(safetensors2param.keys())
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    world_size = dist.get_world_size()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    size = int(math.ceil(total_files / world_size))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ckpt_chunks = [ckpt_chunks[rank * size:rank * size + size] for rank in range(world_size)]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    shard_states = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device = torch.cuda.current_device()
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for rank, files in enumerate(ckpt_chunks):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if rank == dist.get_rank():
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for file in files:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                file = os.path.join(filepath, file)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                states = load_file(file, device=device)
                # print(f"rank {rank} loading {file}...")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                shard_states.update(states)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for file in files:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for param_name in safetensors2param[file]:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    shard_states[param_name] = rank
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return shard_states


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def parallel_init_module_fn(module: torch.nn.Module, shard_states: Dict[str, torch.nn.Parameter]):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Generate a function to initialize sub-modules in the `module` with `shard_states`
    from huggingface checkpoint.

    Args:
        module (torch.nn.Module): the global module to be initialized
        shard_states (Dict[str, torch.nn.Parameter]): the shard states from huggingface checkpoint

    Returns:
        init_fn (Callable): a function to initialize sub-modules in the `module` with `shard_states`
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    state2fqn = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for name, state in itertools.chain(module.named_parameters(remove_duplicate=False),
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       module.named_buffers(remove_duplicate=False)):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state2fqn.setdefault(state, []).append(name)
    # remove standalone parameters and buffers
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    shared = {s for s, names in state2fqn.items() if len(names) > 1}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    materialized_states = {}

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def create_and_sync_state(param_name, state, is_param):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert param_name in shard_states, f"{param_name} not loaded"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device = torch.cuda.current_device()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if is_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:  # buffer
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = torch.empty_like(state.data, device=device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loaded = shard_states[param_name]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(loaded, (torch.nn.Parameter, torch.Tensor)):
            # NOTE: loaded.dtype can be different with param.dtype
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param.data.copy_(loaded.data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(param.data, src=dist.get_rank())
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert isinstance(loaded, int)  # the rank that holds the state
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dist.broadcast(param.data, src=loaded)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shard_states.pop(param_name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        del loaded
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return param

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for name, state in param_and_buffers:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not state.is_meta:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            is_param = name in sub_mod._parameters
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            fqn = state2fqn[state].pop(0)
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (not is_param) and fqn not in shard_states:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if state.is_meta:
                    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                    raise RuntimeError(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "Such buffer is not saved in checkpoint and user should guarantee to init in CPU / GPU device.")
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # for shared parameter, we get it from the first time it is created
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if state in shared:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if state not in materialized_states:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if fqn in shard_states:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        shard_states.pop(fqn)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                materialize_state = materialized_states[state]
            # for not shared parameter, we create it directly
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                materialize_state = create_and_sync_state(fqn, state, is_param)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_param:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sub_mod._parameters[name] = materialize_state
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sub_mod._buffers[name] = materialize_state
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if recurse:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for module in sub_mod.children():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                init_fn(module, recurse=True)

        # for debug
        # if len(shard_states) == 0: print("clear")
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return sub_mod

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return init_fn