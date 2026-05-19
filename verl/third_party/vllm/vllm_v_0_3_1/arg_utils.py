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
import argparse
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import dataclasses
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Optional, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import ModelConfig


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class EngineArgs:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Arguments for vLLM engine."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_hf_config: PretrainedConfig = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dtype: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    kv_cache_dtype: str = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    seed: int = 0
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_model_len: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    worker_use_ray: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pipeline_parallel_size: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor_parallel_size: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_parallel_loading_workers: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    block_size: int = 16
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    swap_space: int = 4  # GiB
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    gpu_memory_utilization: float = 0.90
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_num_batched_tokens: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_num_seqs: int = 256
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_paddings: int = 256
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_log_stats: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    revision: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer_revision: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    quantization: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    load_format: str = 'model'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enforce_eager: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_context_len_to_capture: int = 8192
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_custom_all_reduce: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    enable_lora: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_loras: int = 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_lora_rank: int = 16
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_extra_vocab_size: int = 256
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_dtype = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_cpu_loras: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device: str = 'cuda'

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
    def create_engine_configs(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config = DeviceConfig(self.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config = ModelConfig(self.model_hf_config, self.dtype, self.seed, self.load_format, self.revision,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   self.tokenizer_revision, self.max_model_len, self.quantization, self.enforce_eager,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   self.max_context_len_to_capture)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config = CacheConfig(self.block_size, self.gpu_memory_utilization, self.swap_space, self.kv_cache_dtype,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   model_config.get_sliding_window())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config = ParallelConfig(self.pipeline_parallel_size, self.tensor_parallel_size, self.worker_use_ray,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         self.max_parallel_loading_workers, self.disable_custom_all_reduce)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens, self.max_num_seqs, model_config.max_model_len,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           self.max_paddings)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config = LoRAConfig(max_lora_rank=self.max_lora_rank,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 max_loras=self.max_loras,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 lora_extra_vocab_size=self.lora_extra_vocab_size,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 lora_dtype=self.lora_dtype,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras and self.max_cpu_loras > 0 else
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 None) if self.enable_lora else None
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return (model_config, cache_config, parallel_config, scheduler_config, device_config, lora_config)


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class AsyncEngineArgs(EngineArgs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Arguments for asynchronous vLLM engine."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    engine_use_ray: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    disable_log_requests: bool = False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_log_len: Optional[int] = None

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser = EngineArgs.add_cli_args(parser)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--engine-use-ray',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            action='store_true',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='use Ray to start the LLM engine in a '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'separate process as the server process.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--disable-log-requests', action='store_true', help='disable logging requests')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parser.add_argument('--max-log-len',
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            type=int,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            default=None,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            help='max number of prompt characters or prompt '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'ID numbers being printed in log. '
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'Default: unlimited.')
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return parser
