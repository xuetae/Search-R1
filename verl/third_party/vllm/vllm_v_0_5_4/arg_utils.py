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
import argparse
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import dataclasses
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import ModelConfig, LoadConfig

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, MultiModalConfig,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         ObservabilityConfig, ParallelConfig, PromptAdapterConfig, SchedulerConfig, SpeculativeConfig,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         TokenizerPoolConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.executor.executor_base import ExecutorBase
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import FlexibleArgumentParser
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import str_to_int_tuple

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if TYPE_CHECKING:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (BaseTokenizerGroup)

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def nullable_str(val: str):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not val or val == "None":
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return val


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class EngineArgs:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Arguments for vLLM engine."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_hf_config: PretrainedConfig = None  # for verl
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    served_model_name = None  # TODO(sgm): check this
    # tokenizer: Optional[str] = None # TODO(sgm): check this
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    skip_tokenizer_init: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer_mode: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    trust_remote_code: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    download_dir: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    load_format: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dtype: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    kv_cache_dtype: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    quantization_param_path: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seed: int = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_model_len: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    worker_use_ray: bool = False
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    distributed_executor_backend: Optional[Union[str, Type[ExecutorBase]]] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pipeline_parallel_size: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor_parallel_size: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_parallel_loading_workers: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    block_size: int = 16
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enable_prefix_caching: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_sliding_window: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    use_v2_block_manager: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    swap_space: int = 4  # GiB
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cpu_offload_gb: int = 0  # GiB
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    gpu_memory_utilization: float = 0.90
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_num_batched_tokens: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_num_seqs: int = 256
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_log_stats: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    revision: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    code_revision: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rope_scaling: Optional[dict] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rope_theta: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer_revision: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    quantization: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enforce_eager: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_context_len_to_capture: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_seq_len_to_capture: int = 8192
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_custom_all_reduce: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer_pool_extra_config: Optional[dict] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enable_lora: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_loras: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_lora_rank: int = 16
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enable_prompt_adapter: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_prompt_adapters: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_prompt_adapter_token: int = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    fully_sharded_loras: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_extra_vocab_size: int = 256
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_dtype: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_cpu_loras: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray_workers_use_nsight: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_gpu_blocks_override: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_lookahead_slots: int = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_loader_extra_config: Optional[dict] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ignore_patterns: Optional[Union[str, List[str]]] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    preemption_mode: Optional[str] = None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    scheduler_delay_factor: float = 0.0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enable_chunked_prefill: Optional[bool] = None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    guided_decoding_backend: str = 'outlines'
    # Speculative decoding configuration.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    speculative_model: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    speculative_draft_tensor_parallel_size: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_speculative_tokens: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    speculative_max_model_len: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    speculative_disable_by_batch_size: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ngram_prompt_lookup_max: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ngram_prompt_lookup_min: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    qlora_adapter_name_or_path: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    otlp_traces_endpoint: Optional[str] = None

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        # TODO(shengguangming): delete the unused args
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--model',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default='facebook/opt-125m',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='name or path of the huggingface model to use')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--tokenizer',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.tokenizer,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='name or path of the huggingface tokenizer to use')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--revision',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=None,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='the specific model version to use. It can be a branch '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'name, a tag name, or a commit id. If unspecified, will use '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'the default version.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--tokenizer-revision',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=None,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='the specific tokenizer version to use. It can be a branch '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'name, a tag name, or a commit id. If unspecified, will use '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'the default version.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--tokenizer-mode',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.tokenizer_mode,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            choices=['auto', 'slow'],
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='tokenizer mode. "auto" will use the fast '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'tokenizer if available, and "slow" will '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'always use the slow tokenizer.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--trust-remote-code', action='store_true', help='trust remote code from huggingface')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--download-dir',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.download_dir,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='directory to download and load the weights, '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'default to the default cache dir of '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'huggingface')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--load-format',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.load_format,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            choices=['auto', 'pt', 'safetensors', 'npcache', 'dummy'],
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='The format of the model weights to load. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '"auto" will try to load the weights in the safetensors format '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'and fall back to the pytorch bin format if safetensors format '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'is not available. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '"pt" will load the weights in the pytorch bin format. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '"safetensors" will load the weights in the safetensors format. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '"npcache" will load the weights in pytorch format and store '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'a numpy cache to speed up the loading. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '"dummy" will initialize the weights with random values, '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'which is mainly for profiling.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--dtype',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.dtype,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='data type for model weights and activations. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'The "auto" option will use FP16 precision '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'for FP32 and FP16 models, and BF16 precision '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'for BF16 models.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--max-model-len',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=None,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='model context length. If unspecified, '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'will be automatically derived from the model.')
        # Parallel arguments
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--worker-use-ray',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            action='store_true',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='use Ray for distributed serving, will be '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'automatically set when using more than 1 GPU')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--pipeline-parallel-size',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '-pp',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.pipeline_parallel_size,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='number of pipeline stages')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--tensor-parallel-size',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '-tp',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.tensor_parallel_size,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='number of tensor parallel replicas')
        # KV cache arguments
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--block-size',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.block_size,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            choices=[8, 16, 32],
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--seed', type=int, default=EngineArgs.seed, help='random seed')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--swap-space',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.swap_space,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='CPU swap space size (GiB) per GPU')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--gpu-memory-utilization',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=float,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.gpu_memory_utilization,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='the percentage of GPU memory to be used for'
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'the model executor')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--max-num-batched-tokens',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.max_num_batched_tokens,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='maximum number of batched tokens per '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'iteration')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--max-num-seqs',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=EngineArgs.max_num_seqs,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='maximum number of sequences per iteration')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--disable-log-stats', action='store_true', help='disable logging statistics')
        # Quantization settings.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--quantization',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            '-q',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=str,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            choices=['awq', None],
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=None,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='Method used to quantize the weights')
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return parser

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return engine_args

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def create_engine_config(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> EngineConfig:
        # bitsandbytes quantization needs a specific model loader
        # so we make sure the quant method and the load format are consistent
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (self.quantization == "bitsandbytes" or
           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
           self.qlora_adapter_name_or_path is not None) and \
           self.load_format != "bitsandbytes":
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("BitsAndBytes quantization and QLoRA adapter only support "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"'bitsandbytes' load format, but got {self.load_format}")

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (self.load_format == "bitsandbytes" or
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("BitsAndBytes load format and QLoRA adapter only support "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"'bitsandbytes' quantization, but got {self.quantization}")

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.cpu_offload_gb >= 0, ("CPU offload space must be non-negative"
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          f", but got {self.cpu_offload_gb}")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        multimodal_config = MultiModalConfig()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config = DeviceConfig(self.device)
        # NOTE(sgm): we only modify ModelConfig, other configs are import from vllm
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config = ModelConfig(hf_config=self.model_hf_config,
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
                                   multimodal_config=multimodal_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config = CacheConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            block_size=self.block_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpu_memory_utilization=self.gpu_memory_utilization,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            swap_space=self.swap_space,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_dtype=self.kv_cache_dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sliding_window=model_config.get_sliding_window(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            enable_prefix_caching=self.enable_prefix_caching,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_offload_gb=self.cpu_offload_gb,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config = ParallelConfig(pipeline_parallel_size=self.pipeline_parallel_size,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         tensor_parallel_size=self.tensor_parallel_size,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         worker_use_ray=self.worker_use_ray,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         max_parallel_loading_workers=self.max_parallel_loading_workers,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         disable_custom_all_reduce=self.disable_custom_all_reduce,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         tokenizer_pool_config=TokenizerPoolConfig.create_config(
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             self.tokenizer_pool_size,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             self.tokenizer_pool_type,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             self.tokenizer_pool_extra_config,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         ),
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         ray_workers_use_nsight=self.ray_workers_use_nsight,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         distributed_executor_backend=self.distributed_executor_backend)

        # NOTE[VERL]: Use the world_size set by TORCHRUN
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config.world_size = world_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_model_len = model_config.max_model_len
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_long_context = max_model_len > 32768
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.enable_chunked_prefill is None:
            # If not explicitly set, enable chunked prefill by default for
            # long context (> 32K) models. This is to avoid OOM errors in the
            # initial memory profiling phase.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if use_long_context:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                is_gpu = device_config.device_type == "cuda"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_sliding_window = (model_config.get_sliding_window() is not None)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_spec_decode = self.speculative_model is not None
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                has_seqlen_agnostic_layers = (model_config.contains_seqlen_agnostic_layers(parallel_config))
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if (is_gpu and not use_sliding_window and not use_spec_decode and not self.enable_lora and
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        not self.enable_prompt_adapter and not self.enable_prefix_caching and
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        not has_seqlen_agnostic_layers):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.enable_chunked_prefill = True
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    logger.warning("Chunked prefill is enabled by default for models with "
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   "max_model_len > 32K. Currently, chunked prefill might "
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   "not work with some features or models. If you "
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   "encounter any issues, please disable chunked prefill "
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   "by setting --enable-chunked-prefill=False.")
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.enable_chunked_prefill is None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.enable_chunked_prefill = False

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.enable_chunked_prefill and use_long_context:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.warning(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "The model has a long context length (%s). This may cause OOM "
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "errors during the initial memory profiling phase, or result "
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "in low performance due to small KV cache space. Consider "
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "setting --max-model-len to a smaller value.", max_model_len)

        # TODO: spec config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_model_config=model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_parallel_config=parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            target_dtype=self.dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_model=self.speculative_model,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_draft_tensor_parallel_size = \
                self.speculative_draft_tensor_parallel_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_speculative_tokens=self.num_speculative_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_disable_by_batch_size=self.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_disable_by_batch_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_max_model_len=self.speculative_max_model_len,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            enable_chunked_prefill=self.enable_chunked_prefill,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_v2_block_manager=self.use_v2_block_manager,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            disable_log_stats=self.disable_log_stats,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            draft_token_acceptance_method=\
                self.spec_decoding_acceptance_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            typical_acceptance_sampler_posterior_threshold=self.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            typical_acceptance_sampler_posterior_threshold,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            typical_acceptance_sampler_posterior_alpha=self.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            typical_acceptance_sampler_posterior_alpha,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            disable_logprobs=self.disable_logprobs_during_spec_decoding,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config = SchedulerConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_num_batched_tokens=self.max_num_batched_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_num_seqs=self.max_num_seqs,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_model_len=model_config.max_model_len,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_v2_block_manager=self.use_v2_block_manager,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_lookahead_slots=(self.num_lookahead_slots
                                 # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                                 if speculative_config is None else speculative_config.num_lookahead_slots),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            delay_factor=self.scheduler_delay_factor,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            enable_chunked_prefill=self.enable_chunked_prefill,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            embedding_mode=model_config.embedding_mode,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            preemption_mode=self.preemption_mode,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config = LoRAConfig(max_lora_rank=self.max_lora_rank,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 max_loras=self.max_loras,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 fully_sharded_loras=self.fully_sharded_loras,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 lora_extra_vocab_size=self.lora_extra_vocab_size,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 long_lora_scaling_factors=self.long_lora_scaling_factors,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 lora_dtype=self.lora_dtype,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras and self.max_cpu_loras > 0 else
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 None) if self.enable_lora else None

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.qlora_adapter_name_or_path is not None and \
            self.qlora_adapter_name_or_path != "":
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.model_loader_extra_config is None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.model_loader_extra_config = {}
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_loader_extra_config["qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_config = LoadConfig(
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

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_adapter_config = PromptAdapterConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_prompt_adapters=self.max_prompt_adapters,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        decoding_config = DecodingConfig(guided_decoding_backend=self.guided_decoding_backend)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        observability_config = ObservabilityConfig(otlp_traces_endpoint=self.otlp_traces_endpoint)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (model_config.get_sliding_window() is not None and scheduler_config.chunked_prefill_enabled and
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                not scheduler_config.use_v2_block_manager):
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("Chunked prefill is not supported with sliding window. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "Set --disable-sliding-window to disable sliding window.")

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return EngineConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_config=model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_config=cache_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_config=parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scheduler_config=scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_config=device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            multimodal_config=multimodal_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_config=speculative_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_config=load_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decoding_config=decoding_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            observability_config=observability_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_adapter_config=prompt_adapter_config,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
