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
from vllm.attention import (AttentionMetadata, get_attn_backend)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig)
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
from vllm.sequence import (MultiModalData, SamplerOutput, SequenceData, SequenceGroupMetadata)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import (CudaMemoryProfiler, is_hip, is_pin_memory_available)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner

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
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_config: LoadConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_cache_dtype: Optional[str] = "auto",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vision_language_config: Optional[VisionLanguageConfig] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parallel_config = parallel_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler_config = scheduler_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_config = lora_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.load_config = load_config

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.sliding_window = (model_config.get_sliding_window() if model_config is not None else None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_config = (device_config if device_config is not None else DeviceConfig())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device = self.device_config.device

        # NOTE(sgm): add for verl
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = model  # this will be replaced by get_model()

        # Set after load_model.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_manager: LRUCacheWorkerLoRAManager = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.graph_memory_pool: Optional[Tuple[int, int]] = None  # Set during graph capture.

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_seq_len_to_capture = (self.model_config.max_seq_len_to_capture if self.model_config is not None else 0)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pin_memory = is_pin_memory_available()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.kv_cache_dtype = kv_cache_dtype
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.vision_language_config = vision_language_config

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.attn_backend = get_attn_backend(self.model_config.dtype if model_config is not None else None)

        # Lazy initialization
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.block_size: int  # Set after initial profiling.
        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.graph_block_tables: torch.Tensor  # Set after initial profiling.

        # Set if the backend is flashinfer.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.flashinfer_workspace_buffer: torch.Tensor

    # NOTE(sgm): initialize model using the actor model
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(self) -> None:
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
                                   vision_language_config=self.vision_language_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_memory_usage = m.consumed_memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.info("Loading model weights took %.4f GB", self.model_memory_usage / float(2**30))

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert hasattr(self.model, "supported_lora_modules") and self.model.supported_lora_modules, (
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "Model does not support LoRA")
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert hasattr(self.model, "embedding_modules"), "Model does not have embedding_modules"
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert hasattr(self.model, "embedding_padding_modules"), "Model does not have embedding_padding_modules"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_manager = LRUCacheWorkerLoRAManager(self.scheduler_config.max_num_seqs,
                                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                          self.scheduler_config.max_num_batched_tokens, self.vocab_size,
                                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                          self.lora_config, self.device, self.model.embedding_modules,
                                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                          self.model.embedding_padding_modules)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model = self.lora_manager.create_lora_manager(self.model)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently scaled KV cache is only enabled on ROCm
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.model_config.quantization_param_path is not None:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.model.load_kv_cache_scales(self.model_config.quantization_param_path)
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
        # 中文注释：下一行继续判断其他条件分支。
        elif self.model_config.quantization_param_path is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.warning("KV cache scaling factors provided, "
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           "but the KV cache data type is not FP8. "
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           "KV cache scaling factors will not be used.")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def prepare_input_tensors(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list: List[SequenceGroupMetadata],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata, Set[LoRARequest], LoRAMapping,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               torch.Tensor]:
        # NOTE(sgm): all workers prepare the input in the same way
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prefill_reqs = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        decode_reqs = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq_group_meta in seq_group_metadata_list:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if seq_group_meta.is_prompt:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                prefill_reqs.append(seq_group_meta)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                decode_reqs.append(seq_group_meta)

        # Prepare input tensors.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_positions,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prefill_attn_metadata,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_lens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            query_lens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_index_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_prompt_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_requests,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            multi_modal_input,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            slot_mapping,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ) = self._prepare_prompt(prefill_reqs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_input_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_input_positions,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_attn_metadata,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_lora_index_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_lora_prompt_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_lora_requests,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_slot_mapping,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ) = self._prepare_decode(decode_reqs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list, seq_lens, query_lens, self.device,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     self.pin_memory)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.scheduler_config.chunked_prefill_enabled:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert (len(prefill_reqs) and len(decode_reqs)) == 0

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_prefills = len(seq_lens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_prefill_tokens = len(input_tokens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_decode_tokens = len(decode_input_tokens)

        # Coalesce tensors. Note that attn_metadata is currently not
        # coalesced for simplicity.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_tokens.extend(decode_input_tokens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_positions.extend(decode_input_positions)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        slot_mapping.extend(decode_slot_mapping)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_index_mapping.extend(decode_lora_index_mapping)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_prompt_mapping.extend(decode_lora_prompt_mapping)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_requests.update(decode_lora_requests)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_positions = torch.tensor(input_positions, dtype=torch.long, device=self.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_mapping = LoRAMapping(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                lora_index_mapping,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                lora_prompt_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_mapping = None

        # Broadcast the metadata.
        # If batch contains both prefill and decode, it sends 2 broadcasts.
        # If it only contains 1 type, it triggers a single broadcast.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (prefill_attn_metadata is not None and decode_attn_metadata is not None):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_type = BatchType.MIXED
        # 中文注释：下一行继续判断其他条件分支。
        elif prefill_attn_metadata is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_type = BatchType.PREFILL
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_type = BatchType.DECODE

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attn_metadata = AttentionMetadata(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_prefills=num_prefills,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            slot_mapping=slot_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_prefill_tokens=num_prefill_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_decode_tokens=num_decode_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prefill_metadata=prefill_attn_metadata,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_metadata=decode_attn_metadata,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_cache_dtype=self.kv_cache_dtype,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (input_tokens, input_positions, attn_metadata, sampling_metadata, lora_requests, lora_mapping,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                multi_modal_input)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.inference_mode()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list: List[SequenceGroupMetadata],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_caches: List[torch.Tensor],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Optional[SamplerOutput]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (input_tokens, input_positions, attn_metadata, sampling_metadata, lora_requests, lora_mapping,
         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
         multi_modal_input) = self.prepare_input_tensors(seq_group_metadata_list)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prefill_meta = attn_metadata.prefill_metadata
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        decode_meta = attn_metadata.decode_metadata
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if prefill_meta is None and decode_meta.use_cuda_graph:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            graph_batch_size = input_tokens.shape[0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_executable = self.graph_runners[graph_batch_size]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_executable = self.model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        execute_model_kwargs = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "input_ids": input_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "positions": input_positions,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "kv_caches": kv_caches,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "attn_metadata": attn_metadata,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.vision_language_config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            execute_model_kwargs.update({"image_input": multi_modal_input})
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        # if not self.is_driver_worker:
        #     return None

        # TODO(sgm): perform sampling on rank 0
        # Sample the next token.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = self.model.sample(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logits=logits,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sampling_metadata=sampling_metadata,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output
