# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import EngineConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.engine.arg_utils import EngineArgs

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import LoadConfig, ModelConfig


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class EngineArgs(EngineArgs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_hf_config: PretrainedConfig = None  # for verl

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __post_init__(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def create_model_config(self) -> ModelConfig:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ModelConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hf_config=self.model_hf_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer_mode=self.tokenizer_mode,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            trust_remote_code=self.trust_remote_code,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=self.dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seed=self.seed,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            revision=self.revision,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            code_revision=self.code_revision,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rope_scaling=self.rope_scaling,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rope_theta=self.rope_theta,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer_revision=self.tokenizer_revision,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_model_len=self.max_model_len,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            quantization=self.quantization,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            quantization_param_path=self.quantization_param_path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            enforce_eager=self.enforce_eager,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_context_len_to_capture=self.max_context_len_to_capture,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_logprobs=self.max_logprobs,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            disable_sliding_window=self.disable_sliding_window,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            skip_tokenizer_init=self.skip_tokenizer_init,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            served_model_name=self.served_model_name,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_async_output_proc=not self.disable_async_output_proc,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            override_neuron_config=self.override_neuron_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            config_format=self.config_format,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mm_processor_kwargs=self.mm_processor_kwargs,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def create_load_config(self) -> LoadConfig:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return LoadConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_format=self.load_format,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            download_dir=self.download_dir,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_loader_extra_config=self.model_loader_extra_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ignore_patterns=self.ignore_patterns,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def create_engine_config(self) -> EngineConfig:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        engine_config = super().create_engine_config()

        # NOTE[VERL]: Use the world_size set by torchrun
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        engine_config.parallel_config.world_size = world_size

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return engine_config
