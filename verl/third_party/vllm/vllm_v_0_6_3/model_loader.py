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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""Utilities for selecting and loading models."""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Optional, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PreTrainedModel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import CacheConfig, DeviceConfig, LoadConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader import BaseModelLoader
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader.loader import _initialize_model
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import LoadConfig, LoadFormat, ModelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .dtensor_weight_loaders import load_dtensor_weights, update_dtensor_weight_loader
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .hf_weight_loader import update_hf_weight_loader
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .megatron_weight_loaders import load_megatron_weights, update_megatron_weight_loader


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_model(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    actor_model: Union[PreTrainedModel, Dict],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_config: ModelConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    load_config: LoadConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device_config: DeviceConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parallel_config: ParallelConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    scheduler_config: SchedulerConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    lora_config: Optional[LoRAConfig],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cache_config: CacheConfig = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    loader = get_model_loader(load_config)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format.startswith("dummy"):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return loader.load_model(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_config=model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_config=device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_config=parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scheduler_config=scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_config=cache_config,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return loader.load_model(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_model=actor_model,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_config=model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_config=device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_config=parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scheduler_config=scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_config=cache_config,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Get a model loader based on the load format."""

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(load_config.load_format, type):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return load_config.load_format(load_config)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.AUTO:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_megatron_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return MegatronLoader(load_config)

    # NOTE(sgm): change the weight_loader function in runtime
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.MEGATRON:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_megatron_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return MegatronLoader(load_config)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.HF:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_hf_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return HFLoader(load_config)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.DTENSOR:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_dtensor_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DTensorLoader(load_config)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.DUMMY_HF:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_hf_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DummyModelLoader(load_config)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.DUMMY_MEGATRON:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_megatron_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DummyModelLoader(load_config)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if load_config.load_format == LoadFormat.DUMMY_DTENSOR:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_dtensor_weight_loader()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DummyModelLoader(load_config)

    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError("load format not supported in verl: {}, only support {} and {}".format(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_config.load_format, LoadFormat.MEGATRON, LoadFormat.HF))


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DummyModelLoader(BaseModelLoader):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Model loader that will set model weights to random values."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, load_config: LoadConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(load_config)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_config.model_loader_extra_config:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Model loader extra config is not supported for "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"load format {load_config.load_format}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def download_model(self, model_config: ModelConfig) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        *,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> nn.Module:
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with set_default_torch_dtype(model_config.dtype):
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.device(device_config.device):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            # initialize_dummy_weights(model)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return model.eval()


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronLoader(BaseModelLoader):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Model loader that can load the model weights from partitioned megatron model."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, load_config: LoadConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(load_config)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_config.model_loader_extra_config:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Model loader extra config is not supported for "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"load format {load_config.load_format}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def download_model(self, model_config: ModelConfig) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass  # Nothing to download

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        # NOTE(shengguangming) Load the weights from the actor model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass
        # if isinstance(actor_model, nn.Module):
        #     load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
        # else:
        #     load_weights(actor_weights=actor_model, vllm_model=model)
        # return actor_model

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_model: Union[PreTrainedModel, Dict],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> nn.Module:
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with set_default_torch_dtype(model_config.dtype):
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.device(device_config.device):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)

            # TODO(sgm): This is a hack, we need to register the load_weight() func for each model in vllm
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(actor_model, nn.Module):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_megatron_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)),
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      vllm_model=model)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_megatron_weights(actor_weights=actor_model, vllm_model=model)

            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for _, module in model.named_modules():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                quant_method = getattr(module, "quant_method", None)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if quant_method is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if hasattr(module, "process_weights_after_loading"):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return model.eval()


# 中文注释：下一行定义类，用于组织相关状态与行为。
class HFLoader(BaseModelLoader):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Model loader that can load the model weights from model's full params."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, load_config: LoadConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(load_config)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_config.model_loader_extra_config:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Model loader extra config is not supported for "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"load format {load_config.load_format}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def download_model(self, model_config: ModelConfig) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass  # Nothing to download

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_weights_iterator(self, actor_model: Union[PreTrainedModel, Dict]):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(actor_model, Dict):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return actor_model.items()
        # 中文注释：下一行继续判断其他条件分支。
        elif isinstance(actor_model, nn.Module):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return dict(actor_model.named_parameters()).items()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"actor model should be Dict or nn.Module, but get {type(actor_model)}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_model: Union[PreTrainedModel, Dict],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> nn.Module:
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with set_default_torch_dtype(model_config.dtype):
            # with torch.device(device_config.device):
            # NOTE(sgm): init the model in cpu
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model.load_weights(self._get_weights_iterator(actor_model))
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for _, module in model.named_modules():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                quant_method = getattr(module, "quant_method", None)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if quant_method is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if hasattr(module, "process_weights_after_loading"):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return model.eval()


# 中文注释：下一行定义类，用于组织相关状态与行为。
class DTensorLoader(BaseModelLoader):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Model loader that can load the model weights from partitioned megatron model."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, load_config: LoadConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__(load_config)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_config.model_loader_extra_config:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Model loader extra config is not supported for "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"load format {load_config.load_format}")

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def download_model(self, model_config: ModelConfig) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass  # Nothing to download

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_weights_iterator(actor_model: Union[PreTrainedModel, Dict]):
        # NOTE(shengguangming) Load the weights from the actor model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass
        # if isinstance(actor_model, nn.Module):
        #     load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
        # else:
        #     load_weights(actor_weights=actor_model, vllm_model=model)
        # return actor_model

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_model: Union[PreTrainedModel, Dict],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> nn.Module:
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with set_default_torch_dtype(model_config.dtype):
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.device(device_config.device):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model = _initialize_model(model_config, self.load_config, lora_config, cache_config, scheduler_config)

            # TODO(sgm): This is a hack, we need to register the load_weight() func for each model in vllm
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(actor_model, nn.Module):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_dtensor_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     vllm_model=model)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_dtensor_weights(actor_weights=actor_model, vllm_model=model)

            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for _, module in model.named_modules():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                quant_method = getattr(module, "quant_method", None)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if quant_method is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if hasattr(module, "process_weights_after_loading"):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    module.process_weights_after_loading()
        # NOTE(sgm) Some weights are point to gpu, but still need this.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return model.eval()


# FIXME(sgm): hack the _get_logits function in vllm v0.4.2
# as they use ray, the _get_logits result will only need to return to the driver node,
# therefore gather is enough. However, we use SPMD instead of a central scheduler,
# all_gather is required (aligned with v0.2.6)
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # Get the logits for the next tokens.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logits = torch.matmul(hidden_states, embedding.t())
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if embedding_bias is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits += embedding_bias
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logits = tensor_model_parallel_all_gather(logits)
    # Remove paddings in vocab (if any).
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if logits is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = logits[:, :self.org_vocab_size]
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return logits


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.logits_processor import LogitsProcessor


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def logitsprocessor_init(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vocab_size: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    org_vocab_size: Optional[int] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    scale: float = 1.0,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logits_as_input: bool = False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    soft_cap: Optional[float] = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Args:
        scale: A scaling factor to apply to the logits.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    super(LogitsProcessor, self).__init__()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.scale = scale
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.vocab_size = vocab_size
    # Whether the input is logits (default is hidden states).
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.logits_as_input = logits_as_input
    # original vocabulary size (without LoRA).
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.org_vocab_size = org_vocab_size or vocab_size
    # Soft cap the logits. Used in Gemma 2.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.soft_cap = soft_cap
    # Whether to use gather or all-gather to gather the logits.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.use_gather = False


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
LogitsProcessor.__init__ = logitsprocessor_init  # use all_gather
