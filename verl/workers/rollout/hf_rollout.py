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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import contextlib
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import get_eos_mask
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .base import BaseRollout

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import GenerationConfig

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['HFRollout']


# 中文注释：下一行定义类，用于组织相关状态与行为。
class HFRollout(BaseRollout):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, module: nn.Module, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module = module

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = prompts.batch.batch_size[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_prompts = prompts.chunk(chunks=num_chunks)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = [self._generate_minibatch(p) for p in batch_prompts]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = DataProto.concat(output)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_token_id = prompts.meta_info['eos_token_id']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_token_id = prompts.meta_info['pad_token_id']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = idx.size(0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_length = idx.size(1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.eval()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        do_sample = prompts.meta_info.get('do_sample', self.config.do_sample)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if top_k is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            top_k = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        top_k = max(0, top_k)  # to be compatible with vllm

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        temperature = prompts.meta_info.get('temperature', self.config.temperature)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with param_ctx:
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.module.generate(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    input_ids=idx,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    attention_mask=attention_mask,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    do_sample=do_sample,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    max_new_tokens=response_length,
                    # max_length=max_length,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    eos_token_id=eos_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    pad_token_id=pad_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    generation_config=generation_config,
                    # renormalize_logits=True,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    output_scores=False,  # this is potentially very large
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return_dict_in_generate=True,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    use_cache=True)
        # TODO: filter out the seq with no answers like ds-chat
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq = output.sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sequence_length = prompt_length + self.config.response_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        delta_length = sequence_length - seq.shape[1]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if delta_length > 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            delta_tokens = pad_token_id * delta_tokens
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq = torch.cat((seq, delta_tokens), dim=1)

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert seq.shape[1] == sequence_length

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response = seq[:, prompt_length:]  # (bs, response_length)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = response.size(1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_position_ids = position_ids[:, -1:] + delta_position_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = TensorDict(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'prompts': prompt,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'responses': response,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'input_ids': seq,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'attention_mask': attention_mask,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'position_ids': position_ids
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            },
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size=batch_size)

        # empty cache before compute old_log_prob
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.train()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(batch=batch)
