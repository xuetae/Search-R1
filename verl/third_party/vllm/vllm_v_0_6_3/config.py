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
from dataclasses import dataclass, field
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import TYPE_CHECKING, List, Optional, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig

# Add for verl
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import ModelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import is_hip

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if TYPE_CHECKING:
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from vllm.model_executor.model_loader.loader import BaseModelLoader

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LoadFormat(str, enum.Enum):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    AUTO = "auto"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MEGATRON = "megatron"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    HF = "hf"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DTENSOR = "dtensor"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DUMMY_HF = "dummy_hf"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DUMMY_MEGATRON = "dummy_megatron"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    DUMMY_DTENSOR = "dummy_dtensor"


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ModelConfig(ModelConfig):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(model=hf_config._name_or_path, tokenizer=hf_config._name_or_path, *args, **kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.hf_config = hf_config


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
