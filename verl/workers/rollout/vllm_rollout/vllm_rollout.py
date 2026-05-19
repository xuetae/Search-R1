# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from contextlib import contextmanager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import DictConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.rollout.base import BaseRollout
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.third_party.vllm import LLM, vllm_version
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.third_party.vllm import parallel_state as vllm_ps
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return token_ids


# 中文注释：下一行定义类，用于组织相关状态与行为。
class vLLMRollout(BaseRollout):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import os
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            train_tp = kwargs.get('train_tp', None)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine = LLM(actor_module,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    tokenizer=tokenizer,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    model_hf_config=model_hf_config,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    tensor_parallel_size=tensor_parallel_size,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    dtype=config.dtype,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    enforce_eager=config.enforce_eager,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    skip_tokenizer_init=False,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    max_model_len=config.prompt_length + config.response_length,
                                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                    load_format=config.load_format)

        # Offload vllm model to reduce peak memory usage
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine.offload_model_weights()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kwargs = dict(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            n=1,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            logprobs=1,  # can be set to 0 and let actor to recompute
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_tokens=config.response_length,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # we may detokenize the result all together later
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for k in config.keys():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if hasattr(SamplingParams(), str(k)):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                kwargs[k] = config.get(k)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"kwargs: {kwargs}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.sampling_params = SamplingParams(**kwargs)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.pad_token_id = tokenizer.pad_token_id

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @contextmanager
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_sampling_params(self, **kwargs):
        # update sampling params
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        old_sampling_params_args = {}
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if kwargs:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for key, value in kwargs.items():
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if hasattr(self.sampling_params, key):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    old_value = getattr(self.sampling_params, key)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    old_sampling_params_args[key] = old_value
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    setattr(self.sampling_params, key, value)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, value in old_sampling_params_args.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            setattr(self.sampling_params, key, value)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.free_cache_engine:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.inference_engine.init_cache_engine()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = prompts.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_token_id = prompts.meta_info['eos_token_id']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = idx.size(0)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(batch_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        do_sample = prompts.meta_info.get('do_sample', True)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not do_sample:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kwargs = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'best_of': 1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'top_p': 1.0,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'top_k': -1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'min_p': 0.0,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'temperature': 0,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'n': 1  # if greedy, only 1 response
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }

        # users can customize different sampling_params at different run
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.update_sampling_params(**kwargs):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.inference_engine.generate(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                prompts=None,  # because we have already convert it to prompt token id
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sampling_params=self.sampling_params,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                prompt_token_ids=idx_list,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response = output[0].to(idx.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_probs = output[1].to(idx.device)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if response.shape[1] < self.config.response_length:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.n > 1 and do_sample:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            idx = idx.repeat_interleave(self.config.n, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size = batch_size * self.config.n
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq = torch.cat([idx, response], dim=-1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = response.size(1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_position_ids = position_ids[:, -1:] + delta_position_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = TensorDict(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'prompts': idx,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'responses': response,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'attention_mask': attention_mask,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'position_ids': position_ids
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            },
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size=batch_size)

        # free vllm cache engine
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.free_cache_engine:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.inference_engine.free_cache_engine()

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(batch=batch)
