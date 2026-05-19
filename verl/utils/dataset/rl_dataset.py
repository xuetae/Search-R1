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
from omegaconf import ListConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import pandas as pd

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.utils.data import Dataset, DataLoader
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoTokenizer, PreTrainedTokenizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fs import copy_local_path_from_hdfs

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.model import compute_position_id_with_mask
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.torch_functional as verl_F


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collate_fn(data_list: list[dict]) -> dict:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensors = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensors = {}

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for data in data_list:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, val in data.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(val, torch.Tensor):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if key not in tensors:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    tensors[key] = []
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensors[key].append(val)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if key not in non_tensors:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    non_tensors[key] = []
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                non_tensors[key].append(val)

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in tensors.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors[key] = torch.stack(val, dim=0)

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in non_tensors.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensors[key] = np.array(val, dtype=object)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output.update(tensors)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output.update(non_tensors)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RLHFDataset(Dataset):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 parquet_files: Union[str, List[str]],
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 tokenizer: PreTrainedTokenizer,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 prompt_key='prompt',
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 max_prompt_length=1024,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 filter_prompts=True,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 cache_dir='~/.cache/verl/rlhf',
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 chat_template_func=None,
                 # 中文注释：下一行返回当前函数的计算结果或控制信号。
                 return_raw_chat=False,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 truncation='error'):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not isinstance(parquet_files, (List, ListConfig)):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parquet_files = [parquet_files]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parquet_files = parquet_files
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_dir = os.path.expanduser(cache_dir)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.prompt_key = prompt_key
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.max_prompt_length = max_prompt_length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.filter_prompts = filter_prompts

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.return_raw_chat = return_raw_chat
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.chat_template_func = chat_template_func
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.truncation = truncation

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._download()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._read_files_and_tokenize()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _download(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.fs import copy_local_path_from_hdfs
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, parquet_file in enumerate(self.parquet_files):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

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
        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer = self.tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'filter dataset len: {len(self.dataframe)}')

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __len__(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return len(self.dataframe)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __getitem__(self, item):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_dict = self.dataframe.iloc[item].to_dict()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        chat = row_dict.pop(self.prompt_key)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.tokenizer.chat_template:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                         tokenizer=self.tokenizer,
                                                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                         max_length=self.max_prompt_length,
                                                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                         left_pad=True,
                                                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                         truncation=self.truncation)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = compute_position_id_with_mask(attention_mask)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_dict['input_ids'] = input_ids[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_dict['attention_mask'] = attention_mask[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.return_raw_chat:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        index = row_dict.get("extra_info", {}).get("index", 0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        row_dict["index"] = index

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return row_dict
