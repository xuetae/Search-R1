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
import enum
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import json
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Optional, Union
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass, field, fields

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.quantization import get_quantization_config
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.transformers_utils.config import get_hf_text_config
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import is_hip, print_warning_once
# Add for verl
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len, get_served_model_name

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
GPTQMarlinConfig = get_quantization_config("gptq_marlin")

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_GB = 1 << 30


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ModelConfig(ModelConfig):
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
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use max_seq_len_to_capture instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model 
            names provided, the first name will be used. If not specified, 
            the model name will be the same as `model`.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        hf_config: PretrainedConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_mode: str,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trust_remote_code: bool,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dtype: Union[str, torch.dtype],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seed: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        code_revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rope_scaling: Optional[dict] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rope_theta: Optional[float] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_model_len: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        quantization: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        quantization_param_path: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        enforce_eager: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_context_len_to_capture: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_seq_len_to_capture: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_logprobs: int = 20,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        disable_sliding_window: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        skip_tokenizer_init: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        served_model_name: Optional[Union[str, List[str]]] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        multimodal_config: Optional["MultiModalConfig"] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model = hf_config._name_or_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_config._name_or_path
        # NOTE(sgm): same as open-sourced
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer_mode = tokenizer_mode
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.trust_remote_code = trust_remote_code
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.seed = seed
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.revision = revision
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.code_revision = code_revision
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rope_scaling = rope_scaling
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rope_theta = rope_theta
        # The tokenizer version is consistent with the model version by default.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tokenizer_revision is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tokenizer_revision = revision
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tokenizer_revision = tokenizer_revision
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.quantization = quantization
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.quantization_param_path = quantization_param_path
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.enforce_eager = enforce_eager
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if max_context_len_to_capture is not None:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("`max_context_len_to_capture` is deprecated. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "Use `max_seq_len_to_capture` instead.")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_seq_len_to_capture = max_seq_len_to_capture
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_logprobs = max_logprobs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.disable_sliding_window = disable_sliding_window
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.skip_tokenizer_init = skip_tokenizer_init

        # self.hf_config = get_config(model, trust_remote_code, revision)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hf_config = hf_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hf_text_config = get_hf_text_config(hf_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        # self.served_model_name = get_served_model_name(model,
        #                                                served_model_name)
        # self._verify_load_format()
        # self._verify_tokenizer_mode()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if (not self.disable_sliding_window and self.hf_text_config.model_type == "gemma2" and
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.hf_text_config.sliding_window is not None):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_warning_once("Gemma 2 uses sliding window attention for every odd layer, "
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               "which is currently not supported by vLLM. Disabling sliding "
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               "window and capping the max length to the sliding window size "
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               f"({self.hf_text_config.sliding_window}).")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.disable_sliding_window = True

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_model_len = _get_and_verify_max_len(hf_config=self.hf_text_config,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     max_model_len=max_model_len,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     disable_sliding_window=self.disable_sliding_window,
                                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                     sliding_window_len=self.get_hf_config_sliding_window())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.served_model_name = get_served_model_name(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model,  # str
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            served_model_name)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.multimodal_config = multimodal_config

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.skip_tokenizer_init:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._verify_tokenizer_mode()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_embedding_mode()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_quantization()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_cuda_graph()


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LoadFormat(str, enum.Enum):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    AUTO = 'auto'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON = "megatron"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    HF = "hf"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DTENSOR = 'dtensor'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DUMMY_HF = 'dummy_hf'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DUMMY_MEGATRON = 'dummy_megatron'
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DUMMY_DTENSOR = 'dummy_dtensor'


# TODO: check whether this is necessary
# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class LoadConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
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
            "tensorizer" will use CoreWeave's tensorizer library for
                fast weight loading.
            "bitsandbytes" will load nf4 type weights.
        ignore_patterns: The list of patterns to ignore when loading the model.
            Default to "original/**/*" to avoid repeated loading of llama's 
            checkpoints.
            
    """

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    download_dir: Optional[str] = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ignore_patterns: Optional[Union[List[str], str]] = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __post_init__(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_loader_extra_config = self.model_loader_extra_config or {}
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(model_loader_extra_config, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_load_format()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logger.info("Ignoring the following patterns when downloading weights: %s", self.ignore_patterns)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ignore_patterns = ["original/**/*"]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_load_format(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not isinstance(self.load_format, str):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_format = self.load_format.lower()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.load_format = LoadFormat(load_format)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rocm_not_supported_load_format: List[str] = []
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if is_hip() and load_format in rocm_not_supported_load_format:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rocm_supported_load_format = [
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f for f in LoadFormat.__members__ if (f not in rocm_not_supported_load_format)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ]
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"load format '{load_format}' is not supported in ROCm. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"Supported load formats are "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"{rocm_supported_load_format}")
