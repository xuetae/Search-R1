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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional, Union, ClassVar
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from packaging.version import Version

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.transformers_utils.config import get_config
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import get_cpu_memory, is_hip, get_nvcc_cuda_version

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_GB = 1 << 30


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ModelConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hf_config: PretrainedConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dtype: str,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seed: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_format: str = 'model',
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_model_len: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        quantization: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trust_remote_code: Optional[bool] = True,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        enforce_eager: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_context_len_to_capture: Optional[int] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = hf_config._name_or_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_config._name_or_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.load_format = load_format
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.seed = seed
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.revision = revision
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer_revision = tokenizer_revision
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.quantization = quantization
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.trust_remote_code = trust_remote_code
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.enforce_eager = enforce_eager
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_context_len_to_capture = max_context_len_to_capture

        # self.hf_config = get_config(model, trust_remote_code, revision)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hf_config = hf_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        # self._verify_load_format()
        # self._verify_tokenizer_mode()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_quantization()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_cuda_graph()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_load_format(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_format = self.load_format.lower()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_format not in ["auto", "pt", "safetensors", "npcache", "dummy", "model"]:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Unknown load format: {self.load_format}. Must be one of "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "'auto', 'pt', 'safetensors', 'npcache', 'dummy' or 'model'.")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.load_format = load_format

    # def _verify_tokenizer_mode(self) -> None:
    #     tokenizer_mode = self.tokenizer_mode.lower()
    #     if tokenizer_mode not in ["auto", "slow"]:
    #         raise ValueError(
    #             f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
    #             "either 'auto' or 'slow'.")
    #     self.tokenizer_mode = tokenizer_mode

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_quantization(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        supported_quantization = ["awq", "gptq", "squeezellm"]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rocm_not_supported_quantization = ["awq", "gptq"]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.quantization is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hf_quant_config = getattr(self.hf_config, "quantization_config", None)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hf_quant_config is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hf_quant_method = str(hf_quant_config["quant_method"]).lower()
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.quantization is None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.quantization = hf_quant_method
            # 中文注释：下一行继续判断其他条件分支。
            elif self.quantization != hf_quant_method:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError("Quantization method specified in the model config "
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 f"({hf_quant_method}) does not match the quantization "
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 f"method specified in the `quantization` argument "
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 f"({self.quantization}).")

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.quantization is not None:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.quantization not in supported_quantization:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f"Unknown quantization method: {self.quantization}. Must "
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 f"be one of {supported_quantization}.")
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_hip() and self.quantization in rocm_not_supported_quantization:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f"{self.quantization} quantization is currently not supported "
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 f"in ROCm.")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.warning(f"{self.quantization} quantization is not fully "
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           "optimized yet. The speed can be slower than "
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           "non-quantized models.")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_cuda_graph(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.max_context_len_to_capture is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.max_context_len_to_capture = self.max_model_len
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_context_len_to_capture = min(self.max_context_len_to_capture, self.max_model_len)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (self.quantization in ["gptq", "squeezellm"] and not self.enforce_eager):
            # Related issue: https://github.com/vllm-project/vllm/issues/2147
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.warning(f"{self.quantization} does not support CUDA graph "
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           "yet. Disabling CUDA graph.")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.enforce_eager = True

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def verify_with_parallel_config(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: "ParallelConfig",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_num_attention_heads = self.hf_config.num_attention_heads
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_parallel_size = parallel_config.tensor_parallel_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if total_num_attention_heads % tensor_parallel_size != 0:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Total number of attention heads ({total_num_attention_heads})"
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             " must be divisible by tensor parallel size "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"({tensor_parallel_size}).")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Total number of hidden layers ({total_num_hidden_layers}) "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "must be divisible by pipeline parallel size "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"({pipeline_parallel_size}).")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_sliding_window(self) -> Optional[int]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return getattr(self.hf_config, "sliding_window", None)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_vocab_size(self) -> int:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.hf_config.vocab_size

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_hidden_size(self) -> int:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.hf_config.hidden_size

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_total_num_kv_heads(self) -> int:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_decoder_arch_falcon = (self.hf_config.model_type in falcon_model_types and
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   getattr(self.hf_config, "new_decoder_architecture", False))
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not new_decoder_arch_falcon and getattr(self.hf_config, "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 1

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attributes = [
            # For Falcon:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "n_head_kv",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "num_kv_heads",
            # For LLaMA-2:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "num_key_value_heads",
            # For ChatGLM:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "multi_query_group_num",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for attr in attributes:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_kv_heads = getattr(self.hf_config, attr, None)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if num_kv_heads is not None:
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.hf_config.num_attention_heads

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Returns the number of KV heads per GPU."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


# 中文注释：下一行定义类，用于组织相关状态与行为。
class CacheConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        block_size: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpu_memory_utilization: float,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        swap_space: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_dtype: str,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sliding_window: Optional[int] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.block_size = block_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gpu_memory_utilization = gpu_memory_utilization
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.swap_space_bytes = swap_space * _GB
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_dtype = cache_dtype
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.sliding_window = sliding_window
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_args()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_cache_dtype()

        # Will be set after profiling.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_gpu_blocks = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_cpu_blocks = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_args(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.gpu_memory_utilization > 1.0:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("GPU memory utilization must be less than 1.0. Got "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"{self.gpu_memory_utilization}.")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_cache_dtype(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.cache_dtype == "auto":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pass
        # 中文注释：下一行继续判断其他条件分支。
        elif self.cache_dtype == "fp8_e5m2":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            nvcc_cuda_version = get_nvcc_cuda_version()
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if nvcc_cuda_version < Version("11.8"):
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError("FP8 is not supported when cuda version is lower than 11.8.")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_name = torch.cuda.get_device_name()
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "AMD" in device_name:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise NotImplementedError("FP8_E5M2 KV Cache on AMD GPU has not been supported yet.")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.info("Using fp8_e5m2 data type to store kv cache. It reduces "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "the GPU memory footprint and boosts the performance. "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "But it may cause slight accuracy drop. "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "Currently we only support fp8 without scaling factors and "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "make e5m2 as a default format.")
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def verify_with_parallel_config(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: "ParallelConfig",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpus_per_node = parallel_config.tensor_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        msg = (f"{cpu_memory_usage / _GB:.2f} GiB out of "
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               f"the {total_cpu_memory / _GB:.2f} GiB total CPU memory is "
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               "allocated for the swap space.")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("Too large swap space. " + msg)
        # 中文注释：下一行继续判断其他条件分支。
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.warning("Possibly too large swap space. " + msg)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ParallelConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Whether to use Ray for model workers. Will be set to
            True if either pipeline_parallel_size or tensor_parallel_size is
            greater than 1.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pipeline_parallel_size: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_parallel_size: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        worker_use_ray: bool,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_parallel_loading_workers: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        disable_custom_all_reduce: bool = False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pipeline_parallel_size = pipeline_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tensor_parallel_size = tensor_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker_use_ray = worker_use_ray
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_parallel_loading_workers = max_parallel_loading_workers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.disable_custom_all_reduce = disable_custom_all_reduce

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.world_size = pipeline_parallel_size * tensor_parallel_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.world_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.worker_use_ray = True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_args()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_args(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.pipeline_parallel_size > 1:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError("Pipeline parallelism is not supported yet.")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.disable_custom_all_reduce and self.world_size > 1:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_hip():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.disable_custom_all_reduce = True
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logger.info("Disabled the custom all-reduce kernel because it is not "
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            "supported on AMD GPUs.")
            # 中文注释：下一行继续判断其他条件分支。
            elif self.pipeline_parallel_size > 1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.disable_custom_all_reduce = True
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logger.info("Disabled the custom all-reduce kernel because it is not "
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            "supported with pipeline parallelism.")

        # FIXME(woosuk): Fix the stability issues and re-enable the custom
        # all-reduce kernel.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.disable_custom_all_reduce and self.world_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.disable_custom_all_reduce = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.info("Custom all-reduce kernels are temporarily disabled due to "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "stability issues. We will re-enable them once the issues are "
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        "resolved.")


# 中文注释：下一行定义类，用于组织相关状态与行为。
class SchedulerConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        max_paddings: Maximum number of paddings to be added to a batch.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_num_batched_tokens: Optional[int],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_num_seqs: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_model_len: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_paddings: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if max_num_batched_tokens is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.max_num_batched_tokens = max_num_batched_tokens
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # If max_model_len is too short, use 2048 as the default value for
            # higher throughput.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.max_num_batched_tokens = max(max_model_len, 2048)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_num_seqs = max_num_seqs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_model_len = max_model_len
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_paddings = max_paddings
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_args()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_args(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.max_num_batched_tokens < self.max_model_len:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"smaller than max_model_len ({self.max_model_len}). "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "This effectively limits the maximum sequence length to "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "max_num_batched_tokens and makes vLLM reject longer "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "sequences. Please increase max_num_batched_tokens or "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "decrease max_model_len.")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.max_num_batched_tokens < self.max_num_seqs:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "be greater than or equal to max_num_seqs "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"({self.max_num_seqs}).")


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DeviceConfig:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, device: str = "cuda") -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device = torch.device(device)


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class LoRAConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_lora_rank: int
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_loras: int
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_cpu_loras: Optional[int] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_dtype: Optional[torch.dtype] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_extra_vocab_size: int = 256
    # This is a constant.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_vocab_padding_size: ClassVar[int] = 256

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __post_init__(self):
        # Keep this in sync with csrc/punica/bgmv/bgmv_config.h
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        possible_max_ranks = (8, 16, 32, 64)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        possible_lora_extra_vocab_size = (0, 256, 512)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.max_lora_rank not in possible_max_ranks:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"max_lora_rank ({self.max_lora_rank}) must be one of "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"{possible_max_ranks}.")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"must be one of {possible_lora_extra_vocab_size}.")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.max_loras < 1:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.max_cpu_loras is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.max_cpu_loras = self.max_loras
        # 中文注释：下一行继续判断其他条件分支。
        elif self.max_cpu_loras < self.max_loras:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"max_loras ({self.max_loras})")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def verify_with_model_config(self, model_config: ModelConfig):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.lora_dtype in (None, "auto"):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_dtype = model_config.dtype
        # 中文注释：下一行继续判断其他条件分支。
        elif isinstance(self.lora_dtype, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.lora_dtype = getattr(torch, self.lora_dtype)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_config.quantization is not None:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("LoRA is not supported with quantized models yet.")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if scheduler_config.max_num_batched_tokens > 65528:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("Due to limitations of the custom LoRA CUDA kernel, "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "max_num_batched_tokens must be <= 65528 when "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "LoRA is enabled.")


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_STR_DTYPE_TO_TORCH_DTYPE = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "half": torch.float16,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "float16": torch.float16,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "float": torch.float32,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "float32": torch.float32,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "bfloat16": torch.bfloat16,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_ROCM_NOT_SUPPORTED_DTYPE = ["float", "float32"]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_and_verify_dtype(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config: PretrainedConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dtype: Union[str, torch.dtype],
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config_dtype = getattr(config, "torch_dtype", None)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config_dtype is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config_dtype = torch.float32

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(dtype, str):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dtype = dtype.lower()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if dtype == "auto":
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if config_dtype == torch.float32:
                # Following the common practice, we use float16 for float32
                # models.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                torch_dtype = torch.float16
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                torch_dtype = config_dtype
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f"Unknown dtype: {dtype}")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    # 中文注释：下一行继续判断其他条件分支。
    elif isinstance(dtype, torch.dtype):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = dtype
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError(f"Unknown dtype: {dtype}")

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if is_hip() and torch_dtype == torch.float32:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rocm_supported_dtypes = [
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            k for k, v in _STR_DTYPE_TO_TORCH_DTYPE.items() if (k not in _ROCM_NOT_SUPPORTED_DTYPE)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError(f"dtype \'{dtype}\' is not supported in ROCm. "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         f"Supported dtypes are {rocm_supported_dtypes}")

    # Verify the dtype.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch_dtype != config_dtype:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pass
        # 中文注释：下一行继续判断其他条件分支。
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pass
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch_dtype


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_and_verify_max_len(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hf_config: PretrainedConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_model_len: Optional[int],
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> int:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Get and verify the model's maximum length."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    derived_max_model_len = float("inf")
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    possible_keys = [
        # OPT
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "max_position_embeddings",
        # GPT-2
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "n_positions",
        # MPT
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "max_seq_len",
        # ChatGLM2
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "seq_length",
        # Others
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "max_sequence_length",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "max_seq_length",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "seq_len",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key in possible_keys:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_len_key = getattr(hf_config, key, None)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if max_len_key is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if derived_max_model_len == float("inf"):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return max_model_len

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        default_max_len = 2048
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.warning("The model's config.json does not contain any of the following "
                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                       "keys to determine the original maximum length of the model: "
                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                       f"{possible_keys}. Assuming the model's maximum length is "
                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                       f"{default_max_len}.")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        derived_max_model_len = default_max_len

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if rope_scaling is not None:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert "factor" in rope_scaling
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scaling_factor = rope_scaling["factor"]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if rope_scaling["type"] == "yarn":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            derived_max_model_len = rope_scaling["original_max_position_embeddings"]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        derived_max_model_len *= scaling_factor

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if max_model_len is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_model_len = derived_max_model_len
    # 中文注释：下一行继续判断其他条件分支。
    elif max_model_len > derived_max_model_len:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError(f"User-specified max_model_len ({max_model_len}) is greater than "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         f"the derived max_model_len ({max_len_key}={derived_max_model_len}"
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         " in model's config.json). This may lead to incorrect model "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "outputs or CUDA errors. Make sure the value is correct and "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "within the model context size.")
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return int(max_model_len)
