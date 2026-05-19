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
The main entry point to run the PPO algorithm
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.hdfs_io as hdfs_io
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.torch_functional as verl_F
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import DictConfig, open_dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base import Worker
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.decorator import register, Dispatch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils import hf_tokenizer
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.debug import log_gpu_memory_usage
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fs import copy_local_path_from_hdfs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.import_utils import import_external_libs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.model import compute_position_id_with_mask
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.flops_counter import FlopsCounter
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from codetiming import Timer

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = logging.getLogger(__file__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ActorRolloutRefWorker(Worker):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: DictConfig, role: str):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch.distributed
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not torch.distributed.is_initialized():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = torch.distributed.get_world_size()
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        # build device mesh for Ulysses Sequence Parallel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_device_mesh = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp = world_size // self.ulysses_sequence_parallel_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.ulysses_sequence_parallel_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        mesh_dim_names=['dp', 'sp'])

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.role = role
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_param = False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_grad = False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_optimizer = False
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        # 中文注释：下一行继续判断其他条件分支。
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        self.ulysses_sequence_parallel_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_rollout:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               self.ulysses_sequence_parallel_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_ref:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           self.ulysses_sequence_parallel_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_model_optimizer(self,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               model_path,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               fsdp_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               optim_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               override_model_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               use_remove_padding=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               enable_gradient_checkpointing=False,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               trust_remote_code=False):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.model import print_model_size, update_model_config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.torch_dtypes import PrecisionType
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoModelForCausalLM, AutoConfig
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch import optim

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = fsdp_config.get('model_dtype', None)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch_dtype is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_remove_padding:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.models.registry import check_model_support_rmpad
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            check_model_support_rmpad(actor_model_config.model_type)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            apply_monkey_patch(actor_model_config, verbose=True)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config_kwargs = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'bos_token_id': self.tokenizer.bos_token_id,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'eos_token_id': self.tokenizer.eos_token_id,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pad_token_id': self.tokenizer.pad_token_id,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config_kwargs.update(override_model_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with init_context(), warnings.catch_warnings():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.simplefilter("ignore")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                torch_dtype=torch_dtype,
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                config=actor_model_config,
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                attn_implementation='flash_attention_2',
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_module.to(torch_dtype)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if enable_gradient_checkpointing:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.barrier()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_model_size(actor_module)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if mixed_precision_config is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_dtype = torch.bfloat16
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reduce_dtype = torch.float32
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer_dtype = torch.float32

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_ref:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mixed_precision = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            auto_wrap_policy = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if auto_wrap_policy is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_module_fsdp = FSDP(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_module,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_init_fn=init_fn,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_orig_params=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            auto_wrap_policy=auto_wrap_policy,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_id=torch.cuda.current_device(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sharding_strategy=sharding_strategy,  # zero3
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mixed_precision=mixed_precision,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_module_states=True,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_mesh=self.device_mesh,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            forward_prefetch=False)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          lr=optim_config.lr,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            total_steps = optim_config.get('total_training_steps', 0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                   num_warmup_steps=num_warmup_steps)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_optimizer = None
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_lr_scheduler = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_rollout(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        infer_tp = self.config.rollout.tensor_model_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp = self.world_size // infer_tp
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.rollout.name == 'hf':
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.rollout import HFRollout
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.sharding_manager import BaseShardingManager
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        # 中文注释：下一行继续判断其他条件分支。
        elif self.config.rollout.name == 'vllm':
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  config=self.config.rollout,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  tokenizer=self.tokenizer,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  model_hf_config=self.actor_model_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.distributed.get_world_size() == 1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.config.rollout.load_format = 'dummy_hf'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               inference_engine=rollout.inference_engine,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               model_config=self.actor_model_config,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               device_mesh=rollout_device_mesh)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After building sharding manager', logger=None)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return rollout, rollout_sharding_manager

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        import_external_libs(self.config.model.get('external_lib', None))

        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from omegaconf import OmegaConf
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_remove_padding = self.config.model.get('use_remove_padding', False)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self._is_actor:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config = self.config.actor.optim
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                fsdp_config = self.config.actor.fsdp_config
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config = None
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                fsdp_config = OmegaConf.create()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model_path=self.config.model.path,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                fsdp_config=fsdp_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config=optim_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                override_model_config=override_model_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_remove_padding=use_remove_padding,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                offload_fsdp_grad(module=self.actor_module_fsdp)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self._is_offload_optimizer:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            OmegaConf.set_struct(self.config.actor, True)
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with open_dict(self.config.actor):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.config.actor.use_remove_padding = use_remove_padding
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              actor_module=self.actor_module_fsdp,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              actor_optimizer=self.actor_optimizer)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_rollout:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_ref:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               optim_config=None,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               override_model_config=override_model_config,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               use_remove_padding=use_remove_padding,
                                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                               trust_remote_code=self.config.model.get(
                                                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                   'trust_remote_code', False))[0]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self._is_offload_param:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            OmegaConf.set_struct(self.config.ref, True)
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with open_dict(self.config.ref):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.config.ref.use_remove_padding = use_remove_padding
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.flops_counter = FlopsCounter(self.actor_model_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_actor(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_actor
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_optimizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.cuda()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before update policy', logger=logger)

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.ulysses_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with Timer(name='update_policy', logger=None) as timer:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metrics = self.actor.update_policy(data=data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            delta_time = timer.last
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            global_num_tokens = data.meta_info['global_token_num']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_lr_scheduler.step()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metrics['actor/lr'] = lr

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = DataProto(meta_info={'metrics': metrics})

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = output.to('cpu')

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_optimizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_log_prob(self, data: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """mostly copying from generate_sequences"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_rollout
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.cuda()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info.update(meta_info)

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.ulysses_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            old_log_probs = self.actor.compute_log_prob(data=data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = DataProto.from_dict(tensors={'old_log_probs': old_log_probs})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ulysses_sharding_manager.postprocess_data(output)
            
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output
        
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def generate_sequences(self, prompts: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompts = prompts.to('cuda')
        # set to False if it is validation
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_rollout
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompts.batch = prompts.batch.cuda()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompts.meta_info.update(meta_info)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.rollout_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.rollout.generate_sequences(prompts=prompts)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After rollout generation', logger=logger)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.rollout_sharding_manager.postprocess_data(output)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with self.ulysses_sharding_manager:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.ulysses_sharding_manager.preprocess_data(output)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                old_log_probs = self.actor.compute_log_prob(data=output)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output.batch['old_log_probs'] = old_log_probs
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.ulysses_sharding_manager.postprocess_data(output)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_ref_log_prob(self, data: DataProto):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_ref

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['micro_batch_size'] = micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['temperature'] = self.config.rollout.temperature
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.ulysses_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = self.ulysses_sharding_manager.preprocess_data(data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ref_policy.compute_log_prob(data=data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def save_checkpoint(self, local_path, hdfs_path=None):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_actor
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch.distributed
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict = self.actor.actor_module.state_dict()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Saving actor checkpoint to {local_path}')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.makedirs(local_path, exist_ok=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tokenizer.save_pretrained(local_path)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if hdfs_path is not None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'Uploading actor checkpoint to {hdfs_path}')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.barrier()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class CriticWorker(Worker):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch.distributed
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not torch.distributed.is_initialized():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.init_process_group(backend="nccl")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = torch.distributed.get_world_size()
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.device_mesh import init_device_mesh
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_device_mesh = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp = world_size // self.ulysses_sequence_parallel_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.ulysses_sequence_parallel_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        mesh_dim_names=['dp', 'sp'])

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                  self.ulysses_sequence_parallel_size)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.torch_dtypes import PrecisionType
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch import optim

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from omegaconf import OmegaConf
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config_kwargs = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'bos_token_id': self.tokenizer.bos_token_id,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'eos_token_id': self.tokenizer.eos_token_id,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pad_token_id': self.tokenizer.pad_token_id,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config_kwargs.update(override_config)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Critic overriding config {override_config_kwargs}')

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoConfig, AutoModelForTokenClassification
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch import nn

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trust_remote_code = False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_model_config.num_labels = 1

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_remove_padding = config.model.get('use_remove_padding', False)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_remove_padding:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.models.registry import check_model_support_rmpad
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            check_model_support_rmpad(critic_model_config.model_type)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            apply_monkey_patch(critic_model_config, verbose=True)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_context = get_init_weight_context_manager()
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with init_context(), warnings.catch_warnings():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.simplefilter("ignore")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            setattr(critic_model_config, 'classifier_dropout', 0.)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            setattr(critic_model_config, 'hidden_dropout', '0')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            torch_dtype=torch_dtype,
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            config=critic_model_config,
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            attn_implementation='flash_attention_2',
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            critic_module.to(torch_dtype)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if config.model.get('enable_gradient_checkpointing', False):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_model_size(critic_module)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_model_config = critic_model_config

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        fsdp_config = self.config.model.fsdp_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if mixed_precision_config is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_dtype = torch.bfloat16
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reduce_dtype = torch.float32
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            buffer_dtype = torch.float32

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before critic FSDP', logger=None)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_module = FSDP(critic_module,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             param_init_fn=init_fn,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             use_orig_params=False,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             auto_wrap_policy=auto_wrap_policy,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             device_id=torch.cuda.current_device(),
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             mixed_precision=mixed_precision,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             sync_module_states=True,
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             forward_prefetch=False)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After critic FSDP', logger=None)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       lr=config.optim.lr,
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        total_steps = config.optim.get('total_training_steps', 0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                num_warmup_steps=num_warmup_steps)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return critic_module, critic_optimizer, critic_lr_scheduler

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        import_external_libs(self.config.model.get('external_lib', None))

        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.workers.critic import DataParallelPPOCritic
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_optimizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic = DataParallelPPOCritic(config=self.config,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            critic_module=self.critic_module,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            critic_optimizer=self.critic_optimizer)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.flops_counter = FlopsCounter(self.critic_model_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_values(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.critic_module,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batch_size = self.config.forward_micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['micro_batch_size'] = micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.ulysses_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            values = self.critic.compute_values(data=data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = DataProto.from_dict(tensors={'values': values})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_critic(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.critic_module,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_optimizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.ulysses_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
            with Timer(name='update_critic', logger=None) as timer:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                metrics = self.critic.update_critic(data=data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            delta_time = timer.last

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            global_num_tokens = data.meta_info['global_token_num']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_lr_scheduler.step()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            metrics['critic/lr'] = lr

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = DataProto(batch=None, meta_info={'metrics': metrics})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_optimizer:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def save_checkpoint(self, local_path, hdfs_path=None):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_fsdp_param_and_grad(module=self.critic_module,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     device_id=torch.cuda.current_device(),
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch.distributed
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            state_dict = self.critic_module.state_dict()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Saving critic checkpoint to {local_path}')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.makedirs(local_path, exist_ok=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tokenizer.save_pretrained(local_path)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if hdfs_path is not None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'Uploading critic checkpoint to {hdfs_path}')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.barrier()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


# TODO(sgm): we may need to extract it to dp_reward_model.py
# 中文注释：下一行定义类，用于组织相关状态与行为。
class RewardModelWorker(Worker):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import torch.distributed
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not torch.distributed.is_initialized():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.init_process_group(backend="nccl")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = torch.distributed.get_world_size()
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.device_mesh import init_device_mesh
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_device_mesh = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp = world_size // self.ulysses_sequence_parallel_size
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.ulysses_sequence_parallel_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        mesh_dim_names=['dp', 'sp'])

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.micro_batch_size //= torch.distributed.get_world_size()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_model(self, config):
        # the following line is necessary
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoModelForTokenClassification, AutoConfig
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = copy_local_path_from_hdfs(config.model.path)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.model.input_tokenizer is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._do_switch_chat_template = False
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._do_switch_chat_template = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trust_remote_code = config.model.get('trust_remote_code', False)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config.num_labels = 1

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        use_remove_padding = config.model.get('use_remove_padding', False)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_remove_padding:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.models.registry import check_model_support_rmpad
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            check_model_support_rmpad(model_config.model_type)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with init_context(), warnings.catch_warnings():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.simplefilter("ignore")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            setattr(model_config, 'classifier_dropout', 0.)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            config=model_config,
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            torch_dtype=torch.bfloat16,
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            attn_implementation='flash_attention_2',
                                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                            trust_remote_code=trust_remote_code)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reward_module.to(torch.bfloat16)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_module = FSDP(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            reward_module,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param_init_fn=init_fn,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_orig_params=False,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            auto_wrap_policy=auto_wrap_policy,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_id=torch.cuda.current_device(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sync_module_states=True,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            forward_prefetch=False)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return reward_module

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        import_external_libs(self.config.model.get('external_lib', None))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.reward_module = self._build_model(config=self.config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _forward_micro_batch(self, micro_batch):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids = micro_batch['input_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size, seqlen = input_ids.shape
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            attention_mask = micro_batch['attention_mask']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            position_ids = micro_batch['position_ids']

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.use_remove_padding:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.ulysses_sequence_parallel_size > 1:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            attention_mask=None,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            position_ids=position_ids_rmpad,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            use_cache=False)  # prevent model thinks we are generating
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                reward_rmpad = output.logits
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self.ulysses_sequence_parallel_size > 1:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           gather_dim=0,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           unpad_dim=0,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           padding_size=pad_size)

                # pad it back
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output = self.reward_module(input_ids=input_ids,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            attention_mask=attention_mask,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            position_ids=position_ids)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_score = output.logits  # (batch_size, seq_len, 1)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return rm_score

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask = data.batch['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        position_ids = data.batch['position_ids']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response_length = data.batch['responses'].shape[-1]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        token_level_scores = token_level_scores[:, -response_length:]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return token_level_scores

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _switch_chat_template(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        src_max_length = data.batch['attention_mask'].shape[-1]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        src_tokenizer = self.input_tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        target_tokenizer = self.tokenizer

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_input_ids = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_attention_mask = []

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_ids = data.batch['responses'][i]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response_length = response_ids.shape[-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            response = response.replace(src_tokenizer.eos_token, '')

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            chat.append({'role': 'assistant', 'content': response})

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                             add_generation_prompt=False,
                                                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                             tokenize=False)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.rank == 0 and i == 0:
                # for debugging purpose
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_length = self.config.get('max_length', src_max_length)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if max_length is None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_length = src_max_length
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                prompt=prompt_with_chat_template,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tokenizer=target_tokenizer,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_length=max_length,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pad_token_id=target_tokenizer.pad_token_id,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                left_pad=False,  # right padding
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_input_ids.append(input_ids)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_attention_mask.append(attention_mask)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto.from_dict(rm_inputs)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_rm_score(self, data: DataProto):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import itertools
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._do_switch_chat_template:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_data = self._switch_chat_template(data)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.ulysses_sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            use_dynamic_bsz = self.config.use_dynamic_bsz
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if use_dynamic_bsz:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for micro_batch in micro_batches:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rm_score = self._forward_micro_batch(micro_batch)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                output.append(rm_score)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scores = torch.cat(output, dim=0)  # (batch_size)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if use_dynamic_bsz:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                indices = list(itertools.chain.from_iterable(indices))
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                scores = scores[revert_indices]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output
