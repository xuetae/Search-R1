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
Megatron Reward Model.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from functools import partial
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import logprobs_from_logits
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import logprobs_from_logits, broadcast_dict_tensor, split_dict_tensor_into_batches
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_dtypes import PrecisionType
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.reward_model.base import BasePPORewardModel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron import sequence_parallel as sp_utils
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.pipeline_parallel import get_forward_backward_func


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronRewardModel(BasePPORewardModel):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 config,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 model_config,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 reward_model_module: torch.nn.ModuleList,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 megatron_config,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 sft_tokenizer=None,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 rm_tokenizer=None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.reward_model_module = reward_model_module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.megatron_config = megatron_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device = 'cuda'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.sft_tokenizer = sft_tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rm_tokenizer = rm_tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_different_tokenizer = rm_tokenizer is not None

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.param_offload:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.offload_params_to_cpu()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def re_encode_by_rm_tokenizer(self, data: DataProto) -> DataProto:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.use_different_tokenizer, 're-encode need rm tokenizer not be None!'
        # need to use rm tokenizer to re-generate input_ids, attention_mask and position_ids
        # 1. remove pad for each sequence
        # 2. decode by sft_tokenizer, remove sft system prompts
        # 3. encode by rm_tokenizer with rm system prompts, get rm_input_ids
        # 4. generate attention_mask and position_ids
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = data.batch['input_ids']  # (bs, seq_len)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = data.batch['position_ids']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ori_values = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ori_bs, ori_seqlen = input_ids.size(0), input_ids.size(1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids_for_rm = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask_for_rm = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids_for_rm = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print_decode = True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ori_seqlen = ori_seqlen + 128
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for id, mask in zip(input_ids, attention_mask):
            # 1. remove pad for each sequence
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_zero_indices = torch.nonzero(mask).view(-1)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            begin_pos, end_pos = non_zero_indices[0].item(), non_zero_indices[-1].item()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_id = id[begin_pos:end_pos + 1]
            # 2. decode by sft_tokenizer, remove sft system prompts
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_result = self.sft_tokenizer.decode(valid_id)
            # workaround
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            decode_with_rm_chat = decode_result.replace("<|user|>\n", "[INST] ").replace(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "</s>\n<|assistant|>\n", " [/INST]").replace("</s> \n<|assistant|>\n", " [/INST]") + "</s>"

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"decode_with_rm_chat: {decode_with_rm_chat}")
            
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if print_decode and torch.distributed.get_rank() == 0:
                # only print first decode result
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'device {torch.cuda.current_device()}: sft decode result:\n{decode_result}\n \
                        \ndevice {torch.cuda.current_device()}: sft decode result with rm chat template:\n{decode_with_rm_chat}\n\n'
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     )
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print_decode = False
            # 3. encode by rm_tokenizer
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_input_ids = self.rm_tokenizer(decode_with_rm_chat,
                                             # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                             return_tensors='pt')['input_ids'][0].to(input_ids.device)
            # 4. generate attention_mask and position_ids
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_attention_mask = torch.ones_like(rm_input_ids, device=input_ids.device)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cur_seqlen = rm_input_ids.shape[-1]
            # NOTE(gh): the later reward compute will process the shape (bs, seqlen_pad_128)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if cur_seqlen > ori_seqlen:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'warninig: rm encode seqlen {cur_seqlen} > sft encode seqlen {ori_seqlen}')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_input_ids = rm_input_ids[:ori_seqlen]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_attention_mask = rm_attention_mask[:ori_seqlen]
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # right padding
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_input_ids = pad_sequence_to_length(rm_input_ids, ori_seqlen, self.rm_tokenizer.pad_token_id)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_attention_mask = pad_sequence_to_length(rm_attention_mask, ori_seqlen, 0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_position_ids = torch.arange(0, ori_seqlen, device=input_ids.device)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids_for_rm.append(torch.unsqueeze(rm_input_ids, dim=0))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask_for_rm.append(torch.unsqueeze(rm_attention_mask, dim=0))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids_for_rm.append(torch.unsqueeze(rm_position_ids, dim=0))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids_for_rm = torch.cat(input_ids_for_rm, dim=0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask_for_rm = torch.cat(attention_mask_for_rm, dim=0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids_for_rm = torch.cat(position_ids_for_rm, dim=0)

        # (bs, seqlen) will not change, but input_ids, attention_mask and position_ids will change
        # NOTE(gh): need to replace into origin values after compute reward!
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['input_ids'] = input_ids_for_rm
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['attention_mask'] = attention_mask_for_rm
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['position_ids'] = position_ids_for_rm

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data, ori_values

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.no_grad()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_reward(self, data: DataProto) -> DataProto:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.param_offload:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.load_params_to_cuda()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_different_tokenizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data, ori_values = self.re_encode_by_rm_tokenizer(data)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = data.batch['input_ids']  # (bs, seq_len')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = data.batch['position_ids']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses = data.batch['responses']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = responses.size(0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = responses.size(1)

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.no_grad():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.forward_batch(data)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits = torch.cat([o['logits'] for o in output], dim=0)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                logits = torch.empty(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    (input_ids.shape[0], input_ids.shape[1]),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dtype=torch.bfloat16,  # TODO(sgm): check why is bfloat16
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    device=input_ids.device)
            # broadcast across pp ranks
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.broadcast(tensor=logits,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        group=mpu.get_pipeline_model_parallel_group(),
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        async_op=False)

        # (bs, seqlen', hidden_size) -> (bs, seqlen', 1) -> (bs, seqlen')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_rewards = logits
        # find the last token reward
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)  # (bs, 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rewards = torch.gather(token_level_rewards, dim=1, index=ends)  # (bs, 1)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.use_different_tokenizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.batch.update(ori_values)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = ori_values['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = ori_values['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = ori_values['position_ids']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_rewards = rewards.expand(attention_mask.shape[0], attention_mask.shape[1])  # (bs, ori_seqlen)

        # assign last valid token reward to ori position
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bs,)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_mask = torch.zeros_like(attention_mask)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_rewards = token_level_rewards * eos_mask
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_rewards = token_level_rewards[:, -response_length:]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.param_offload:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.offload_params_to_cpu()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # add empty cache after each compute
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.empty_cache()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = TensorDict({'rm_scores': token_level_rewards}, batch_size=input_ids.shape[0])

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(batch=batch)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward_batch(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.contiguous()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        broadcast_dict_tensor(data.batch,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              group=mpu.get_pipeline_model_parallel_group())

        # split into micro-batches
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config is not None and 'ppo_micro_batch_size' in self.config:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            infer_batch_size = self.config.ppo_micro_batch_size
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            infer_batch_size = data.batch.batch_size[0]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batches = split_dict_tensor_into_batches(data.batch, batch_size=infer_batch_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        n_micro_batch = len(batches)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_len = batches[0]['input_ids'].shape[1]

        # compute input shapes for pp stages
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_shapes = compute_transformers_input_shapes(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batches,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info={
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'sequence_parallel': self.megatron_config.sequence_parallel,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'hidden_size': self.model_config.hidden_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })
        # compute input shapes for pp stages
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        forward_backward_func = get_forward_backward_func()

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def loss_func(output):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 1., {'logits': output.logits}

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def forward_step(batch_iter, model):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch = next(batch_iter)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = batch['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = batch['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = batch['position_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return output, loss_func

        # batch should be a list of batches inside micro-batches
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_generator = make_batch_generator(batches, vpp_size=len(self.reward_model_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            losses_reduced = forward_backward_func(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_step_func=forward_step,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_iterator=batch_generator,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model=self.reward_model_module,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_microbatches=n_micro_batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_shapes=input_shapes,  # must set for flash-attn sequence packing
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_length=infer_batch_size * seq_len,  # no use when input_shapes was set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_size=self.model_config.hidden_size,  # no use when input_shapes was set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batch_size=1,  # no use when input_shapes was set
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_only=True,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            losses_reduced = forward_backward_func(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_step_func=forward_step,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data_iterator=batch_generator,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model=self.reward_model_module,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_microbatches=n_micro_batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_length=infer_batch_size * seq_len,  # in use for pp = 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hidden_size=self.model_config.hidden_size,  # in use for pp = 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batch_size=1,  # in use for pp = 1
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                forward_only=True,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # loss_reduces contains the stats returned from loss_func

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return losses_reduced

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def offload_params_to_cpu(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device == 'cuda':
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for reward_model_module in self.reward_model_module:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for name, param in reward_model_module.named_parameters():
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    param.data = param.data.to('cpu', non_blocking=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.device = 'cpu'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.empty_cache()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_params_to_cuda(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device == 'cpu':
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for reward_model_module in self.reward_model_module:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for name, param in reward_model_module.named_parameters():
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    param.data = param.data.to(torch.cuda.current_device(), non_blocking=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.device = 'cuda'
