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
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig, Qwen2Config, LlamaConfig

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
VALID_CONFIG_TYPE = (Qwen2Config, LlamaConfig)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_device_flops(unit="T"):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def unit_convert(number, level):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        units = ["B", "K", "M", "G", "T", "P"]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if number <= 0:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return number
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ptr = 0
        # 中文注释：下一行开始循环，直到条件不再满足。
        while ptr < len(units) and units[ptr] != level:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            number /= 1000
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ptr += 1
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return number

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device_name = torch.cuda.get_device_name()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    flops = float("inf")  # INF flops for unkown gpu type
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if "H100" in device_name or "H800" in device_name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops = 989e12
    # 中文注释：下一行继续判断其他条件分支。
    elif "A100" in device_name or "A800" in device_name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops = 312e12
    # 中文注释：下一行继续判断其他条件分支。
    elif "L40" in device_name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops = 181.05e12
    # 中文注释：下一行继续判断其他条件分支。
    elif "L20" in device_name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops = 119.5e12
    # 中文注释：下一行继续判断其他条件分支。
    elif "H20" in device_name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops = 148e12
    # 中文注释：下一行继续判断其他条件分支。
    elif "910B" in device_name:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops = 354e12
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    flops_unit = unit_convert(flops, unit)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return flops_unit


# 中文注释：下一行定义类，用于组织相关状态与行为。
class FlopsCounter:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: PretrainedConfig):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not isinstance(config, VALID_CONFIG_TYPE):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"Only support config type of {VALID_CONFIG_TYPE}, but got {type(config)}. "
                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                  f"MFU will always be zero.")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.estimate_func = {"qwen2": self._estimate_qwen2_flops, 'llama': self._estimate_qwen2_flops}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return 0

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(self.config, (Qwen2Config, LlamaConfig))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_size = self.config.hidden_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vocab_size = self.config.vocab_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_hidden_layers = self.config.num_hidden_layers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_key_value_heads = self.config.num_key_value_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_attention_heads = self.config.num_attention_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        intermediate_size = self.config.intermediate_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        head_dim = hidden_size // num_attention_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        q_size = num_attention_heads * head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        k_size = num_key_value_heads * head_dim
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mlp_N = hidden_size * intermediate_size * 3
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seqlen_square_sum = 0
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seqlen in batch_seqlens:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seqlen_square_sum += seqlen * seqlen
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops_all_token = dense_N_flops + attn_qkv_flops
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return flops_achieved

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def estimate_flops(self, batch_seqlens, delta_time):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokens_sum = sum(batch_seqlens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        promised_flops = get_device_flops()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return estimated_flops, promised_flops
