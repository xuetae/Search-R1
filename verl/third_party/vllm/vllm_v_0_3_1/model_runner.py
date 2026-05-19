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
from typing import Dict, List, Optional, Tuple, Set, Union
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import contextlib
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig, SchedulerConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor import InputMetadata, SamplingMetadata
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sampling_params import SamplingParams, SamplingType
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.layers import LoRAMapping
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import in_wsl
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner, _async_h2d

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .model_loader import get_model

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
KVCache = Tuple[torch.Tensor, torch.Tensor]
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_PAD_SLOT_ID = -1
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
LORA_WARMUP_RANK = 8
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


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
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_cache_dtype: Optional[str] = "auto",
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

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.sliding_window = (model_config.get_sliding_window() if model_config is not None else None)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_config = (device_config if device_config is not None else DeviceConfig())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device = self.device_config.device

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = model  # this will be replaced by get_model()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.block_size = None  # Set after initial profiling.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_manager = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.graph_memory_pool = None  # Set during graph capture.

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_context_len_to_capture = (self.model_config.max_context_len_to_capture
                                           # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                                           if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.in_wsl = in_wsl()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.kv_cache_dtype = kv_cache_dtype

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = get_model(actor_model=self.model,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               model_config=self.model_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               device_config=self.device_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               lora_config=self.lora_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vocab_size = self.model.config.vocab_size

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert hasattr(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.model,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "supported_lora_modules") and self.model.supported_lora_modules, "Model does not support LoRA"
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert hasattr(self.model, "embedding_modules"), "Model does not have embedding_modules"
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert hasattr(self.model, "embedding_padding_modules"), "Model does not have embedding_padding_modules"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_manager = LRUCacheWorkerLoRAManager(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler_config.max_num_seqs,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler_config.max_num_batched_tokens + self.scheduler_config.max_paddings, vocab_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.lora_config, self.device, self.model.embedding_modules, self.model.embedding_padding_modules)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model = self.lora_manager.create_lora_manager(self.model)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _prepare_sample(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list: List[SequenceGroupMetadata],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_lens: List[int],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        subquery_lens: Optional[List[int]],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> SamplingMetadata:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        selected_token_indices: List[int] = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        selected_token_start_idx = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        categorized_sample_indices = {t: [] for t in SamplingType}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        categorized_sample_indices_start_idx = 0

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_subquery_len = max(subquery_lens) if subquery_lens else 1
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_ids = list(seq_group_metadata.seq_data.keys())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sampling_params = seq_group_metadata.sampling_params
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_groups.append((seq_ids, sampling_params))

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if seq_group_metadata.is_prompt:
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert len(seq_ids) == 1
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert subquery_lens is not None
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                subquery_len = subquery_lens[i]
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    categorized_sample_indices_start_idx += subquery_len - 1

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                categorized_sample_indices[sampling_params.sampling_type].append(categorized_sample_indices_start_idx)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                categorized_sample_indices_start_idx += 1

                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if sampling_params.prompt_logprobs is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    selected_token_indices.extend(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        range(selected_token_start_idx, selected_token_start_idx + subquery_len - 1))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                selected_token_indices.append(selected_token_start_idx + subquery_len - 1)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                selected_token_start_idx += max_subquery_len
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_seqs = len(seq_ids)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                selected_token_indices.extend(range(selected_token_start_idx, selected_token_start_idx + num_seqs))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                selected_token_start_idx += num_seqs

                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                categorized_sample_indices[sampling_params.sampling_type].extend(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    range(categorized_sample_indices_start_idx, categorized_sample_indices_start_idx + num_seqs))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                categorized_sample_indices_start_idx += num_seqs

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        selected_token_indices = _async_h2d(selected_token_indices,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            dtype=torch.long,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            target_device=self.device,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            pin_memory=not self.in_wsl)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        categorized_sample_indices = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            t: _async_h2d(seq_ids, dtype=torch.int, target_device=self.device, pin_memory=not self.in_wsl)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for t, seq_ids in categorized_sample_indices.items()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_data: Dict[int, SequenceData] = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq_group_metadata in seq_group_metadata_list:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_data.update(seq_group_metadata.seq_data)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sampling_metadata = SamplingMetadata(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_groups=seq_groups,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_data=seq_data,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_lens=prompt_lens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            selected_token_indices=selected_token_indices,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            categorized_sample_indices=categorized_sample_indices,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return sampling_metadata

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def prepare_input_tensors(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata, Set[int], LoRAMapping]:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if is_prompt:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            (input_tokens, input_positions, input_metadata, prompt_lens, subquery_lens, lora_index_mapping,
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             lora_prompt_mapping, lora_requests) = self._prepare_prompt(seq_group_metadata_list)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            (input_tokens, input_positions, input_metadata, lora_index_mapping, lora_prompt_mapping,
             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
             lora_requests) = self._prepare_decode(seq_group_metadata_list)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_lens = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            subquery_lens = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sampling_metadata = self._prepare_sample(seq_group_metadata_list, prompt_lens, subquery_lens)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            flat_lora_index_mapping = [item for sublist in lora_index_mapping for item in sublist]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_mapping = LoRAMapping(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                flat_lora_index_mapping,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                lora_prompt_mapping,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_mapping = None

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (input_tokens, input_positions, input_metadata, sampling_metadata, lora_requests, lora_mapping)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.inference_mode()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Optional[SamplerOutput]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (input_tokens, input_positions, input_metadata, sampling_metadata, lora_requests,
         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
         lora_mapping) = self.prepare_input_tensors(seq_group_metadata_list)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.set_active_loras(lora_requests, lora_mapping)

        # Execute the model.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if input_metadata.use_cuda_graph:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            graph_batch_size = input_tokens.shape[0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_executable = self.graph_runners[graph_batch_size]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_executable = self.model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hidden_states = model_executable(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids=input_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            positions=input_positions,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_caches=kv_caches,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_metadata=input_metadata,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # Sample the next token.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = self.model.sample(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hidden_states=hidden_states,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sampling_metadata=sampling_metadata,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.inference_mode()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vocab_size = self.model_config.get_vocab_size()
        # FIXME(sgm): this sampling params will call cumsum(), causing the
        # deterministic cumsum throw error
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dummy_lora_requests = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dummy_lora_requests_per_seq = []
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_config:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for idx in range(self.lora_config.max_loras):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                lora_id = idx + 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dummy_lora_request = LoRARequest(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    lora_name=f"warmup_{lora_id}",
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    lora_int_id=lora_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    lora_local_path="/not/a/real/path",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.lora_manager.add_dummy_lora(dummy_lora_request, rank=LORA_WARMUP_RANK)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dummy_lora_requests.append(dummy_lora_request)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dummy_lora_requests_per_seq = [
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dummy_lora_requests[idx % len(dummy_lora_requests)] for idx in range(max_num_seqs)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seqs: List[SequenceGroupMetadata] = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for group_id in range(max_num_seqs):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_len = (max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_data = SequenceData([0] * seq_len)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq = SequenceGroupMetadata(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                request_id=str(group_id),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                is_prompt=True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_data={group_id: seq_data},
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sampling_params=sampling_params,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                block_tables=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                lora_request=dummy_lora_requests_per_seq[group_id] if dummy_lora_requests_per_seq else None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seqs.append(seq)

        # Run the model with the dummy inputs.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_caches = [(None, None)] * num_layers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.execute_model(seqs, kv_caches)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.synchronize()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return
