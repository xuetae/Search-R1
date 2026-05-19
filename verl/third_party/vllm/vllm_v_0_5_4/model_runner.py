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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from enum import IntEnum
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, List, Optional, Set, Tuple, Union
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import vllm.envs as envs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.attention import (AttentionMetadata, get_attn_backend)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         SchedulerConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.layers import LoRAMapping
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor import SamplingMetadata
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.models.interfaces import (supports_lora, supports_vision)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import (CudaMemoryProfiler, is_hip, is_pin_memory_available)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.prompt_adapter.worker_manager import (LRUCacheWorkerPromptAdapterManager)

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .model_loader import get_model
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import ModelConfig, LoadConfig

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)


# How batches are constructed.
# 中文注释：下一行定义类，用于组织相关状态与行为。
class BatchType(IntEnum):
    # Every batch is prefill.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    PREFILL = 0
    # Every batch is decode.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DECODE = 1
    # Batch is a mixture of prefill and decode.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MIXED = 2


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ModelRunner(ModelRunner):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model: Union[nn.Module, Dict], # [verl] model itself or its parameter dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_config: LoadConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_cache_dtype: Optional[str] = "auto",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_driver_worker: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        multimodal_config: Optional[MultiModalConfig] = None,
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return_hidden_states: bool = False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_cache_dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            is_driver_worker=True,  # a hack
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_adapter_config=prompt_adapter_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            multimodal_config=multimodal_config,
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return_hidden_states=return_hidden_states)

        # NOTE(sgm): add for verl
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = model  # this will be replaced by get_model()

    # NOTE(sgm): initialize model using the actor model
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.info("Starting to load model %s...", self.model_config.model)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with CudaMemoryProfiler() as m:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model = get_model(actor_model=self.model,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   model_config=self.model_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   device_config=self.device_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   lora_config=self.lora_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   load_config=self.load_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   parallel_config=self.parallel_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   scheduler_config=self.scheduler_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   multimodal_config=self.multimodal_config,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   cache_config=self.cache_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_memory_usage = m.consumed_memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.info("Loading model weights took %.4f GB", self.model_memory_usage / float(2**30))

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert supports_lora(self.model), "Model does not support LoRA"
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert not supports_vision(self.model), "To be tested: vision language model with LoRA settings."

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_manager = LRUCacheWorkerLoRAManager(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler_config.max_num_seqs,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler_config.max_num_batched_tokens,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.vocab_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.lora_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.device,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.model.embedding_modules,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.model.embedding_padding_modules,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_position_embeddings=self.model.config.max_position_embeddings,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model = self.lora_manager.create_lora_manager(self.model)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.prompt_adapter_config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.prompt_adapter_manager = LRUCacheWorkerPromptAdapterManager(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler_config.max_num_seqs, self.scheduler_config.max_num_batched_tokens, self.device,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.prompt_adapter_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model = (self.prompt_adapter_manager.create_prompt_adapter_manager(self.model))

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.model_config.quantization_param_path is not None:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    warnings.warn(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "Loading kv cache scaling factor from JSON is "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "deprecated and will be removed. Please include "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "kv cache scaling factors in the model checkpoint.",
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        FutureWarning,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        stacklevel=2)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.model.load_kv_cache_scales(self.model_config.quantization_param_path)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    logger.info("Loaded KV cache scaling factors from %s", self.model_config.quantization_param_path)
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                    raise RuntimeError(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "Using FP8 KV cache and scaling factors provided but "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "model %s does not support loading scaling factors.", self.model.__class__)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logger.warning("Using FP8 KV cache but no scaling factors "
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               "provided. Defaulting to scaling factors of 1.0. "
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               "This may lead to less accurate results!")

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if envs.VLLM_TEST_DYNAMO_GRAPH_CAPTURE:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")
