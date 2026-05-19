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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
os.environ['NCCL_DEBUG'] = 'WARN'
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn, optim
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.utils.data import DataLoader, DistributedSampler

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.dataset import SFTDataset
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fs import copy_local_path_from_hdfs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.tracking import Tracking

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.device_mesh import DeviceMesh

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.hdfs_io as hdfs_io
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.debug import log_gpu_memory_usage
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from peft import LoraConfig, TaskType, get_peft_model

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = logging.getLogger(__file__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def extract_step(path):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    match = re.search(r'global_step_(\d+)', path)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if match:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return int(match.group(1))
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return None


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def convert_to_regular_types(obj):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Convert Hydra configs and other special types to regular Python types."""
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from omegaconf import ListConfig, DictConfig
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if isinstance(obj, (ListConfig, DictConfig)):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    # 中文注释：下一行继续判断其他条件分支。
    elif isinstance(obj, (list, tuple)):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return [convert_to_regular_types(x) for x in obj]
    # 中文注释：下一行继续判断其他条件分支。
    elif isinstance(obj, dict):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return obj


# 中文注释：下一行定义类，用于组织相关状态与行为。
class FSDPSFTTrainer(object):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config, device_mesh: DeviceMesh):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_mesh = device_mesh
        # build tokenizer first
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils import hf_tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.data.chat_template is not None:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._normalize_config_bsz()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._build_dataloader()
        # build model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(self.config)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _normalize_config_bsz(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp_size = self.device_mesh.size()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Normalize batch size by dp {dp_size}')

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.config.data.train_batch_size % dp_size == 0
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.config.data.micro_batch_size % dp_size == 0

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.data.train_batch_size //= dp_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.data.micro_batch_size //= dp_size

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_dataloader(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config = self.config
        # build dataset
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        tokenizer=self.tokenizer,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        prompt_key=config.data.prompt_key,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        response_key=config.data.response_key,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        max_length=config.data.max_length,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        truncation=config.data.truncation)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      tokenizer=self.tokenizer,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      prompt_key=config.data.prompt_key,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      response_key=config.data.response_key,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      response_dict_keys=config.data.get('response_dict_keys', None),
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      max_length=config.data.max_length,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      truncation=config.data.truncation)

        # build dataloader
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = self.device_mesh.get_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = self.device_mesh.size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                shuffle=True,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                num_replicas=world_size,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                rank=rank,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                drop_last=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           batch_size=config.data.train_batch_size,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           sampler=self.train_sampler,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           num_workers=8,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           pin_memory=True,
                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                           drop_last=True)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.val_sampler = DistributedSampler(self.val_dataset,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              shuffle=True,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              num_replicas=world_size,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              rank=rank,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              drop_last=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         batch_size=config.data.micro_batch_size,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         sampler=self.val_sampler,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         num_workers=8,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         pin_memory=True,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         drop_last=True)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import importlib
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            importlib.import_module(self.config.model.external_lib)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before model allocation', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)

        # This may be very large
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings)

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with init_context():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                               config=config,
                                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                               torch_dtype=torch.float32,
                                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                               attn_implementation='flash_attention_2',
                                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                               trust_remote_code=trust_remote_code)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.model.get('lora_rank', 0) > 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                lora_config = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'task_type': TaskType.CAUSAL_LM,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'r': self.config.model.lora_rank,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'lora_alpha': self.config.model.lora_alpha,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    'bias': "none"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.model.enable_gradient_checkpointing:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After model allocation', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         reduce_dtype=torch.float32,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         buffer_dtype=torch.float32)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(auto_wrap_policy)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.config.model.fsdp_config.cpu_offload:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_offload = None
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.fsdp_model = FSDP(module=self.model,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               auto_wrap_policy=auto_wrap_policy,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               param_init_fn=init_fn,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               mixed_precision=mixed_precision,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               device_mesh=self.device_mesh,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               sync_module_states=True,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               device_id=torch.cuda.current_device(),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               cpu_offload=cpu_offload,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               use_orig_params=False)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     lr=self.config.optim.lr,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     betas=self.config.optim.betas,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     weight_decay=self.config.optim.weight_decay)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        steps_per_epoch = len(self.train_dataloader)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_steps = steps_per_epoch * self.config.trainer.total_epochs

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'Number of steps/epoch {steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {total_steps}'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_warmup_steps = int(total_steps * self.config.optim.warmup_steps_ratio)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            num_warmup_steps=num_warmup_steps,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            num_training_steps=total_steps)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _compute_loss(self, batch):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        labels = batch['input_ids'][:, 1:].cuda()

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.fsdp_model(input_ids=batch['input_ids'],
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     attention_mask=batch['attention_mask'],
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     position_ids=batch['position_ids'],
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     use_cache=False)  # prevent model thinks it it generating

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logits = output.logits

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shift_logits = logits[..., :-1, :].contiguous()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shift_labels = labels.contiguous()
        # Flatten the tokens
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        shift_labels = shift_labels.to(shift_logits.device)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loss = loss_fct(shift_logits, shift_labels)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loss = loss * loss_mask

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        valid_token_this_rank = torch.sum(loss_mask)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.data.balance_dp_token:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dp_size = torch.distributed.get_world_size()
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dp_size = 1

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        loss = torch.sum(loss) / valid_token_this_rank * dp_size  # possible bugs here for dp
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return loss

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def training_step(self, batch: TensorDict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.fsdp_model.train()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.optimizer.zero_grad()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batches = batch.split(self.config.data.micro_batch_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        n_micro_batches = len(micro_batches)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        step_loss = 0
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for micro_batch in micro_batches:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            loss = self._compute_loss(batch=micro_batch) / n_micro_batches
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            loss.backward()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            step_loss += loss.item()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before optimizer step', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.optimizer.step()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After optimizer step', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lr_scheduler.step()

        # reduce loss across dp ranks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lr = self.lr_scheduler.get_last_lr()[0]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After offload weights', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        step_loss = torch.tensor(step_loss).cuda()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def validation_step(self, batch: TensorDict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.fsdp_model.eval()
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.no_grad():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            loss = self._compute_loss(batch)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return loss

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def save_checkpoint(self, step):
        # save checkpoint
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict = self.fsdp_model.state_dict()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.makedirs(path, exist_ok=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model.save_pretrained(path, state_dict=state_dict)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tokenizer.save_pretrained(path)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.trainer.default_hdfs_dir:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.barrier()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def fit(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                experiment_name=self.config.trainer.experiment_name,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                default_backend=self.config.trainer.logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.trainer.total_training_steps is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_training_steps = self.config.trainer.total_training_steps

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.total_training_steps = total_training_steps
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Total training steps: {self.total_training_steps}')

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.trainer.validate_before_training:
            # validate before training
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            val_losses = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for data in self.val_dataloader:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                val_loss = self.validation_step(data)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                val_losses.append(val_loss)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if rank == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                val_loss = torch.mean(torch.stack(val_losses))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metric = {'val/loss': val_loss.detach().item()}
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tracking.log(data=metric, step=global_step)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.barrier()

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for epoch in range(self.config.trainer.total_epochs):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.train_sampler.set_epoch(epoch=epoch)
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for data in self.train_dataloader:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metric = self.training_step(data)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if rank == 0:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    tracking.log(data=metric, step=global_step)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                global_step += 1

                # for early exit validation
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    val_losses = []
                    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                    for val_data in self.val_dataloader:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size).cuda()
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        val_loss = self.validation_step(val_data)
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        val_losses.append(val_loss)
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if rank == 0:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        metric = {'val/loss': avg_val_loss.detach().item()}
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        tracking.log(data=metric, step=global_step)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    torch.distributed.barrier()

                    # Save final checkpoint
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.save_checkpoint(step=global_step)
                    # 中文注释：下一行返回当前函数的计算结果或控制信号。
                    return

            # validation
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            val_losses = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for data in self.val_dataloader:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                val_loss = self.validation_step(data)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                val_losses.append(val_loss)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if rank == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                val_loss = torch.mean(torch.stack(val_losses))
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metric = {'val/loss': val_loss.detach().item()}
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tracking.log(data=metric, step=global_step)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.barrier()

            # save checkpoint
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.save_checkpoint(step=global_step)


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import hydra

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.device_mesh import init_device_mesh

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.distributed import initialize_global_process_group


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def main(config):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_rank, rank, world_size = initialize_global_process_group()

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('dp',))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    trainer.fit()


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == '__main__':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    main()
