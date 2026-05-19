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
Generate responses given a dataset of prompts
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import hydra
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
os.environ['NCCL_DEBUG'] = 'WARN'
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.model import compute_position_id_with_mask

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import pandas as pd

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoTokenizer

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fs import copy_local_path_from_hdfs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.fsdp_workers import ActorRolloutRefWorker
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.hdfs_io import makedirs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@hydra.main(config_path='config', config_name='generation', version_base=None)
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def main(config):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from pprint import pprint
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from omegaconf import OmegaConf
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    OmegaConf.resolve(config)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_path = copy_local_path_from_hdfs(config.model.path)
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.utils import hf_tokenizer
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer = hf_tokenizer(local_path)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.rollout.temperature == 0.:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dataset = pd.read_parquet(config.data.path)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    chat_lst = dataset[config.data.prompt_key].tolist()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    chat_lst = [chat.tolist() for chat in chat_lst]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer.padding_side = 'left'
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tokenizer.pad_token is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer.pad_token = tokenizer.eos_token

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    wg.init_model()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    config_batch_size = config.data.batch_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_batch = (total_samples // config_batch_size) + 1
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_lst = [[] for _ in range(config.data.n_samples)]

    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for batch_idx in range(num_batch):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               add_generation_prompt=True,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               padding=True,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               truncation=True,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               max_length=config.rollout.prompt_length,
                                               # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                               return_tensors='pt',
                                               # 中文注释：下一行返回当前函数的计算结果或控制信号。
                                               return_dict=True,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               tokenize=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids = inputs['input_ids']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = inputs['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = compute_position_id_with_mask(attention_mask)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = DataProto.from_dict(batch_dict)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        real_batch_size = data.batch['input_ids'].shape[0]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if real_batch_size % dp_size != 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dummy_data_size = dp_size - real_batch_size % dp_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dummy_data = data[:dummy_data_size]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = DataProto.concat([data, dummy_data])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = data.batch['input_ids'].shape[0]
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(config.data.n_samples):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = wg.generate_sequences(data)
            # remove dummy data
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = output[:real_batch_size]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                 skip_special_tokens=False)

            # remove the padding
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pad_token = tokenizer.pad_token
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output_text_unpad = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for text in output_text:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output_text_unpad.append(text.replace(pad_token, ''))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output_lst[i].extend(output_text_unpad)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_lst = np.array(output_lst, dtype=object)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dataset[f'responses'] = output_lst

    # write to a new parquet
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_dir = os.path.dirname(config.data.output_path)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    makedirs(output_dir, exist_ok=True)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dataset.to_parquet(config.data.output_path)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output_text


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    main()
