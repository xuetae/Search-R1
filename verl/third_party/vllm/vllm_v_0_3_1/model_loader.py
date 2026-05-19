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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""Utilities for selecting and loading models."""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import contextlib
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Type, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig, PreTrainedModel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.tensor_parallel.utils import VocabUtility

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.models import ModelRegistry
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.weight_utils import (get_quant_config, initialize_dummy_weights)

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import ModelConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import DeviceConfig, LoRAConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .weight_loaders import *
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sequence import SamplerOutput
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Optional
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.sampler import Sampler
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.sampler import _prune_hidden_states, _apply_logits_processors, _apply_penalties, _apply_top_k_top_p, _apply_min_p, _apply_penalties, _sample, _get_logprobs, _build_sampler_output


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@contextlib.contextmanager
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _set_default_torch_dtype(dtype: torch.dtype):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Sets the default torch dtype to the given dtype."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    old_dtype = torch.get_default_dtype()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.set_default_dtype(dtype)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    yield
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.set_default_dtype(old_dtype)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    architectures = getattr(config, "architectures", [])
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arch in architectures:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_cls = ModelRegistry.load_model_cls(arch)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_cls is not None:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return model_cls
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.linear import *
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.activation import ScaledActivation

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__LAYER_WEIGHT_LOADER_REGISTRY__ = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ColumnParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    MergedColumnParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    QKVParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    RowParallelLinear: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    VocabParallelEmbedding: parallel_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ParallelLMHead: parallel_weight_loader
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): latest commit in vllm fix awq for this function and add load_weights
    # "default_weight_loader": default_weight_loader
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}

# NOTE(gmsheng): change the weight_loader function in runtime
# 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
for layer_class, weight_loader in __LAYER_WEIGHT_LOADER_REGISTRY__.items():
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    layer_class.weight_loader = weight_loader

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__MODEL_WEIGHT_LOADER_REGISTRY__ = {
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'GPT2LMHeadModel': gpt2_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'LlamaForCausalLM': llama_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'LLaMAForCausalLM': llama_weight_loader,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    'MistralForCausalLM': mistral_weight_loader,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
}

# FIXME(shengguangming): the vLLM vocab will pad to 64, which may incur out of bounds
# so we need to rewrite the init function of vocab
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
DEFAULT_VOCAB_PADDING_SIZE = 64


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def vocab_init(self,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               num_embeddings: int,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               embedding_dim: int,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               params_dtype: Optional[torch.dtype] = None,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               org_num_embeddings: Optional[int] = None,
               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
               padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    super(VocabParallelEmbedding, self).__init__()

    # Keep the input dimensions.
    # TODO (pad to be divided by 4)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.num_embeddings = num_embeddings
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.org_vocab_size = org_num_embeddings or num_embeddings

    # self.num_embeddings_padded = pad_vocab_size(num_embeddings,
    #                                             padding_size)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.embedding_dim = embedding_dim
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if params_dtype is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params_dtype = torch.get_default_dtype()
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.tp_size = get_tensor_model_parallel_world_size()
    # Divide the weight matrix along the vocaburaly dimension.

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.vocab_start_index, self.vocab_end_index = (VocabUtility.vocab_range_from_global_vocab_size(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_embeddings, get_tensor_model_parallel_rank(), self.tp_size))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.num_embeddings_per_partition = (self.vocab_end_index - self.vocab_start_index)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self.weight = Parameter(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.empty(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.num_embeddings_per_partition,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.embedding_dim,
            # device=torch.cuda.current_device(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=params_dtype))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    set_weight_attrs(self.weight, {"parallel_dim": 0, "weight_loader": self.weight_loader})


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
VocabParallelEmbedding.__init__ = vocab_init


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_model_weight_loader(arch: str):
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if arch in __MODEL_WEIGHT_LOADER_REGISTRY__:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return __MODEL_WEIGHT_LOADER_REGISTRY__[arch]
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_model(actor_model: Union[PreTrainedModel, Dict],
              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
              model_config: ModelConfig,
              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
              device_config: DeviceConfig,
              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
              lora_config: Optional[LoRAConfig] = None) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the quantization config.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    linear_method = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    quant_config = None
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if model_config.quantization is not None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        quant_config = get_quant_config(model_config.quantization, model_config.model, model_config.hf_config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        model_config.download_dir)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        capability = torch.cuda.get_device_capability()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        capability = capability[0] * 10 + capability[1]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if capability < quant_config.get_min_capability():
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"The quantization method {model_config.quantization} is not "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "supported for the current GPU. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"Minimum capability: {quant_config.get_min_capability()}. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"Current capability: {capability}.")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        supported_dtypes = quant_config.get_supported_act_dtypes()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_config.dtype not in supported_dtypes:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"{model_config.dtype} is not supported for quantization "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"method {model_config.quantization}. Supported dtypes: "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"{supported_dtypes}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        linear_method = quant_config.get_linear_method()

    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        # with torch.device(device_config.device):
        # NOTE(sgm): init the model in cpu
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model_class(model_config.hf_config, linear_method)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_config.load_format == "dummy":
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model = model.cuda()
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            initialize_dummy_weights(model)
        # 中文注释：下一行继续判断其他条件分支。
        elif model_config.load_format == 'model' or model_config.load_format == 'auto':
            # NOTE(shengguangming) Load the weights from the actor model
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(actor_model, nn.Module):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_weights(actor_weights=actor_model, vllm_model=model)

        # NOTE(sgm) Some weights are point to gpu, but still need this.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return model.eval()


# the actor model is .state_dict()
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_weights(actor_weights: Dict, vllm_model: nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    vllm_model = vllm_model.cuda()


# FIXME(sgm): hack the Sampler function in vllm v0.3.1
# as they use ray, the sampler result will only need to return to the driver node,
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


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def forward(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    self,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    embedding: torch.Tensor,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hidden_states: torch.Tensor,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sampling_metadata: SamplingMetadata,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    embedding_bias: Optional[torch.Tensor] = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> Optional[SamplerOutput]:
    # Get the hidden states that we use for sampling.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

    # Get the logits for the next tokens.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logits = self._get_logits(hidden_states, embedding, embedding_bias)
    # save origin logprobs for sampler_output
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    origin_logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # Only perform sampling in the driver worker.
    # Note: `_get_logits` is still distributed across TP workers because
    # the `embedding` weight is distributed across TP workers.
    # TODO(zhuohan): Change the get_logits part to a separate stage.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if not sampling_metadata.perform_sampling:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None

    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert logits is not None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _, vocab_size = logits.shape

    # Apply logits processors (if any).
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logits = _apply_logits_processors(logits, sampling_metadata)

    # Prepare sampling tensors with pinned memory to avoid blocking.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    (sampling_tensors, do_penalties, do_top_p_top_k,
     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
     do_min_p) = SamplingTensors.from_sampling_metadata(sampling_metadata, vocab_size, logits.device, logits.dtype)

    # Apply presence and frequency penalties.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_penalties:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = _apply_penalties(logits, sampling_tensors.prompt_tokens, sampling_tensors.output_tokens,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  sampling_tensors.presence_penalties, sampling_tensors.frequency_penalties,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  sampling_tensors.repetition_penalties)

    # Apply temperature scaling.
    # Use in-place division to avoid creating a new tensor.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_top_p_top_k:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps, sampling_tensors.top_ks)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if do_min_p:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = _apply_min_p(logits, sampling_tensors.min_ps)

    # We use float32 for probabilities and log probabilities.
    # Compute the probabilities.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    # Compute the log probabilities.
    # Use log_softmax to ensure numerical stability.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # Sample the next tokens.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sample_results = _sample(probs, logprobs, sampling_metadata)

    # Get the logprobs query results.
    # prompt_logprobs, sample_logprobs = _get_logprobs(
    #     logprobs, sampling_metadata, sample_results)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt_logprobs, sample_logprobs = _get_logprobs(origin_logprobs, sampling_metadata, sample_results)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _build_sampler_output(sample_results, sampling_metadata, prompt_logprobs, sample_logprobs)


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.layers.sampler import Sampler

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
Sampler._get_logits = _get_logits
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
Sampler.forward = forward
