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
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .sequence_parallel import pad_to_sequence_parallel


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_transformers_input_shapes(batches, meta_info):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from flash_attn.bert_padding import unpad_input  # flash 2 is a must for Megatron
    # pre-compute input shapes for each micro-batch at each pp stage
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_shapes = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for model_inputs in batches:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = model_inputs['input_ids']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = model_inputs['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if meta_info['sequence_parallel']:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            # compute shapes for model_inputs
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_shapes.append(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                torch.Size([
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    input_ids_rmpad.shape[0] // mpu.get_tensor_model_parallel_world_size(), 1, meta_info['hidden_size']
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ]))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # compute shapes for model_inputs
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info['hidden_size']]))
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return input_shapes


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def make_batch_generator(batches, vpp_size):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if vpp_size > 1:
        # has vpp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_generator = [iter(b) for b in batch_generator]
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # no vpp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_generator = iter(batches)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return batch_generator
