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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from contextlib import contextmanager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload, Dict, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tqdm import tqdm
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import PretrainedConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .arg_utils import EngineArgs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .llm_engine_sp import LLMEngine
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm import LLM
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.inputs import (PromptInputs, TextPrompt, TokensPrompt, parse_and_batch_prompt)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.guided_decoding import (GuidedDecodingRequest, get_local_guided_decoding_logits_processor)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.pooling_params import PoolingParams
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.prompt_adapter.request import PromptAdapterRequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sampling_params import SamplingParams
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.transformers_utils.tokenizer import get_cached_tokenizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.usage.usage_lib import UsageContext
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import Counter, deprecate_kwargs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.nn.utils.rnn import pad_sequence
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LLM(LLM):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: A HuggingFace Transformers model instance.
        tokenizer: A HuggingFace Transformers tokenizer instance.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_hf_config: PretrainedConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_mode: str = "auto",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trust_remote_code: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        skip_tokenizer_init: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_parallel_size: int = 1,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dtype: str = "auto",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        quantization: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_revision: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seed: int = 0,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpu_memory_utilization: float = 0.9,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        swap_space: int = 4,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cpu_offload_gb: float = 0,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        enforce_eager: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_context_len_to_capture: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_seq_len_to_capture: int = 8192,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        disable_custom_all_reduce: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_format = 'auto',
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        **kwargs,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if "disable_log_stats" not in kwargs:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kwargs["disable_log_stats"] = True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        engine_args = EngineArgs(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_hf_config=model_hf_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_parallel_size=tensor_parallel_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dtype=dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            quantization=quantization,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            revision=revision,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer_revision=tokenizer_revision,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seed=seed,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpu_memory_utilization=gpu_memory_utilization,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            swap_space=swap_space,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_offload_gb=cpu_offload_gb,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            enforce_eager=enforce_eager,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_context_len_to_capture=max_context_len_to_capture,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_seq_len_to_capture=max_seq_len_to_capture,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            disable_custom_all_reduce=disable_custom_all_reduce,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_format=load_format,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            skip_tokenizer_init=skip_tokenizer_init,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            **kwargs,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not isinstance(tokenizer, tokenizer_cls):
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "one of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.llm_engine = LLMEngine.from_engine_args(model, tokenizer, engine_args)  # TODO: check usagecontext
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.request_counter = Counter()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_cache_engine(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.llm_engine.init_cache_engine()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def free_cache_engine(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.llm_engine.free_cache_engine()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.llm_engine.tokenizer

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def set_tokenizer(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.llm_engine.tokenizer = tokenizer

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_tqdm:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_requests = self.llm_engine.get_num_unfinished_requests()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pbar = tqdm(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                total=num_requests,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                desc="Processed prompts",
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dynamic_ncols=True,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         f"output: {0:.2f} toks/s"),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # Run the engine.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_in_toks = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_out_toks = 0
        # 中文注释：下一行开始循环，直到条件不再满足。
        while self.llm_engine.has_unfinished_requests():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            step_outputs = self.llm_engine.step()
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for output in step_outputs:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if output.finished:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    outputs.append(output)
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if use_tqdm:
                        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            total_in_toks += len(output.prompt_token_ids)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            pbar.postfix = (f"est. speed input: {in_spd:.2f} toks/s, "
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            f"output: {out_spd:.2f} toks/s")
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        pbar.update(1)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_tqdm:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._post_process_outputs(outputs)

    # # NOTE(shengguangming): add for verl
    # # TODO(sgm): we can optimize it by making the dataloader yield List[int] without padding.
    # def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
    #     # remove the left padding in the prompt token_id
    #     pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    #     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    #     token_ids = prompt_token_ids[non_pad_index:].tolist()
    #     return token_ids

    # NOTE(shengguangming): add for verl
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_token_ids = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logprobs = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for request_output in request_outputs:  # List[RequestOutput]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            outputs = request_output.outputs
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for output in outputs:  # List[CompletionOutput], usually len == 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output_token_ids.append(torch.tensor(output.token_ids))
                # TODO(shengguangming): can be optimzied by rewrite the Sampler._get_logprobs() logits
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logprobs_dicts = output.logprobs
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if logprobs_dicts is not None:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    logprob = []
                    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                    for logprobs_dict, id in zip(logprobs_dicts, output.token_ids):
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        logprob.append(logprobs_dict[id].logprob)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    logprobs.append(torch.tensor(logprob))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(logprobs) > 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output_token_ids, logprobs

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.llm_engine.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def offload_model_weights(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.llm_engine.offload_model_weights()
