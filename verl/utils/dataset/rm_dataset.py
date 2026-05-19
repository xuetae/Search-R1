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

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import pandas as pd

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.utils.data import Dataset
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoTokenizer

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils import hf_tokenizer


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def download_files_distributed(download_fn):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import torch.distributed
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.is_initialized():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # download files
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            download_fn()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.barrier()
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # download anyway
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        download_fn()


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RMDataset(Dataset):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 parquet_files: Union[str, List[str]],
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 tokenizer,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 prompt_key='prompt',
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 chosen_key='chosen',
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 rejected_key='rejected',
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 max_length=1024,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 add_eos=True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 cache_dir='~/.cache/verl/rm'):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not isinstance(parquet_files, List):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parquet_files = [parquet_files]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parquet_files = parquet_files
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_dir = os.path.expanduser(cache_dir)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if isinstance(tokenizer, str):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer = hf_tokenizer(tokenizer)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.prompt_key = prompt_key
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.chosen_key = chosen_key
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rejected_key = rejected_key

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.add_eos = add_eos
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_length = max_length

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._download()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._read_files_and_tokenize()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _download(self):

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def _download_files():
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.fs import copy, _is_non_local
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.makedirs(self.cache_dir, exist_ok=True)
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert os.path.exists(self.cache_dir)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i, parquet_file in enumerate(self.parquet_files):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if _is_non_local(parquet_file):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if not os.path.exists(dst):
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        copy(src=parquet_file, dst=dst)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.parquet_files[i] = dst

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        download_files_distributed(_download_files)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _read_files_and_tokenize(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dataframes = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dataframe = pd.read_parquet(parquet_file)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dataframes.append(dataframe)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.dataframe = pd.concat(dataframes)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.prompts = self.dataframe[self.prompt_key].tolist()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __len__(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return len(self.prompts)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _pad_to_length(self, input_ids, attention_mask):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        curr_length = input_ids.shape[-1]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if curr_length < self.max_length:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = torch.cat(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                (input_ids, torch.zeros(size=(self.max_length - curr_length,), dtype=input_ids.dtype)), dim=-1)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = torch.cat(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                (attention_mask, torch.zeros(size=(self.max_length - curr_length,), dtype=attention_mask.dtype)),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dim=-1)
        # 中文注释：下一行继续判断其他条件分支。
        elif curr_length > self.max_length:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = input_ids[:self.max_length]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = attention_mask[:self.max_length]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return input_ids, attention_mask

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __getitem__(self, item):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt = self.prompts[item]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chosen_response = self.chosen_responses[item]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rejected_response = self.rejected_responses[item]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'][0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chosen_response_ids = self.tokenizer(chosen_response, return_tensors='pt')['input_ids'][0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rejected_response_ids = self.tokenizer(rejected_response, return_tensors='pt')['input_ids'][0]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.add_eos:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chosen_response_ids = torch.cat((chosen_response_ids, torch.tensor([self.tokenizer.eos_token_id])), dim=-1)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rejected_response_ids = torch.cat((rejected_response_ids, torch.tensor([self.tokenizer.eos_token_id])),
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              dim=-1)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chosen_input_ids = torch.cat((prompt_ids, chosen_response_ids), dim=-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chosen_attention_mask = torch.ones_like(chosen_input_ids)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rejected_input_ids = torch.cat((prompt_ids, rejected_response_ids), dim=-1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chosen_input_ids, chosen_attention_mask = self._pad_to_length(chosen_input_ids, chosen_attention_mask)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rejected_input_ids, rejected_attention_mask = self._pad_to_length(rejected_input_ids, rejected_attention_mask)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = torch.stack((chosen_input_ids, rejected_input_ids), dim=0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = torch.stack((rejected_input_ids, rejected_attention_mask), dim=0)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'input_ids': input_ids,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'attention_mask': attention_mask,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }