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

import os
from typing import List, Union

import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """Minimal prompt/response SFT dataset used by fsdp_sft_trainer.py.

    The parquet file must contain a prompt column and a response column. The prompt can
    be either a plain string or a chat list in the same format used by the RL datasets.
    Loss is applied only to response tokens.
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key="prompt",
        prompt_dict_keys=None,
        response_key="response",
        response_dict_keys=None,
        max_length=1024,
        truncation="error",
        cache_dir="~/.cache/verl/sft",
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.prompt_dict_keys = prompt_dict_keys
        self.response_key = response_key
        self.response_dict_keys = response_dict_keys
        self.max_length = max_length
        self.truncation = truncation

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy, _is_non_local

        os.makedirs(self.cache_dir, exist_ok=True)
        for i, parquet_file in enumerate(self.parquet_files):
            if _is_non_local(parquet_file):
                dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                if not os.path.exists(dst):
                    copy(src=parquet_file, dst=dst)
                self.parquet_files[i] = dst

    def _read_files(self):
        dataframes = [pd.read_parquet(parquet_file) for parquet_file in self.parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f"SFT dataset len: {len(self.dataframe)}")

    def __len__(self):
        return len(self.dataframe)

    def _format_prompt(self, prompt):
        if self.prompt_dict_keys is not None:
            prompt = {key: prompt[key] for key in self.prompt_dict_keys}

        if isinstance(prompt, list):
            if self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            return "\n".join(item.get("content", "") for item in prompt) + "\nAssistant: "

        return str(prompt)

    def _format_response(self, response):
        if self.response_dict_keys is not None:
            response = {key: response[key] for key in self.response_dict_keys}
        return str(response)

    def _truncate_or_pad(self, input_ids, attention_mask, loss_mask):
        cur_len = input_ids.shape[-1]
        if cur_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"SFT sample length {cur_len} exceeds max_length={self.max_length}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            else:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]

        pad_len = self.max_length - input_ids.shape[-1]
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)])

        return input_ids, attention_mask, loss_mask

    def __getitem__(self, item):
        row = self.dataframe.iloc[item].to_dict()
        prompt = self._format_prompt(row[self.prompt_key])
        response = self._format_response(row[self.response_key])

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        response_ids = self.tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if self.tokenizer.eos_token_id is not None:
            response_ids = torch.cat([response_ids, torch.tensor([self.tokenizer.eos_token_id])])

        input_ids = torch.cat([prompt_ids, response_ids])
        attention_mask = torch.ones_like(input_ids)

        # The trainer shifts labels by one position, so supervise from the token
        # immediately before the response through the EOS token.
        loss_mask = torch.zeros_like(input_ids)
        response_start = max(prompt_ids.shape[-1] - 1, 0)
        loss_mask[response_start:] = 1

        input_ids, attention_mask, loss_mask = self._truncate_or_pad(input_ids, attention_mask, loss_mask)

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "loss_mask": loss_mask.float(),
        }
