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
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import DictConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.megatron.worker import MegatronWorker
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.actor.megatron_actor import MegatronPPOActor
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.critic.megatron_critic import MegatronPPOCritic
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.sharding_manager import AllGatherPPModel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.single_controller.base.decorator import register, Dispatch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.fs import copy_local_path_from_hdfs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.debug import log_gpu_memory_usage
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.model import load_megatron_model_weights
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron_utils import init_model_parallel_config
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron_utils import offload_megatron_param_and_grad, load_megatron_param_and_grad
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils import hf_tokenizer

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import ModelParallelConfig

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = logging.getLogger(__file__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def set_random_seed(seed):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import torch
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import numpy as np
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import random
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.manual_seed(seed)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    np.random.seed(seed)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    random.seed(seed)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.cuda.device_count() > 0:
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core import tensor_parallel
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# 中文注释：下一行定义类，用于组织相关状态与行为。
class ActorRolloutRefWorker(MegatronWorker):
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

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not torch.distributed.is_initialized():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank = int(os.environ['LOCAL_RANK'])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.init_process_group(backend="nccl")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_device(rank)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.actor.megatron.sequence_parallel:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mpu.initialize_model_parallel(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                virtual_pipeline_model_parallel_size=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pipeline_model_parallel_split_rank=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_sharp=False,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                context_parallel_size=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                expert_model_parallel_size=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                nccl_communicator_config_path=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_random_seed(seed=self.config.actor.megatron.seed)

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

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_param = False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_grad = False
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._is_offload_optimizer = False

        # normalize config
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor and self._is_rollout:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_param = self.config.actor.get('param_offload', False)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_grad = self.config.actor.get('grad_offload', False)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_optimizer = self.config.actor.get('optimizer_offload', False)
        # 中文注释：下一行继续判断其他条件分支。
        elif self._is_ref:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._is_offload_param = self.config.ref.get('param_offload', False)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_model_optimizer(self,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               model_path,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               megatron_config: ModelParallelConfig,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               optim_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               override_model_config,
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               enable_gradient_checkpointing=False):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core.models.gpt.gpt_model import ModelType
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.model import print_model_size, update_model_config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = copy_local_path_from_hdfs(model_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the actor_model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_model_config = AutoConfig.from_pretrained(local_path)

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

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def megatron_actor_model_provider(pre_process, post_process):
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.model import get_parallel_model_from_config
            # vpp is not supported yet because it will hang for some reason. Need debugging
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_model = get_parallel_model_from_config(config=actor_model_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            megatron_config=megatron_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            pre_process=pre_process,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            post_process=post_process,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            value=False)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_model.cuda()
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return parallel_model

        # Step 3: initialize the megatron model
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor and self._is_rollout:
            # Initialize the 3D HybridEngine
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            hybrid_engine = AllGatherPPModel(model_provider=megatron_actor_model_provider)
            # Fetch the model at current rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_module = hybrid_engine.this_rank_models
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(actor_module, nn.ModuleList):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                actor_module = [actor_module[0]]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.actor.load_weight:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_megatron_model_weights(self.config,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            actor_model_config,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            actor_module,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            params_dtype=megatron_config.params_dtype,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            is_value_model=False)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.rank == 0:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print_model_size(actor_module[0])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After AllGatherPPModel init', logger=logger)
        # 中文注释：下一行继续判断其他条件分支。
        elif self._is_ref:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'self.config.ref.load_weight: {self.config.ref.load_weight}')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ref_module = get_model(model_provider_func=megatron_actor_model_provider,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   model_type=ModelType.encoder_or_decoder,
                                   # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                   wrap_with_ddp=False)
            # ref_module = nn.ModuleList(ref_module)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.ref.load_weight:  # should align with the actor:
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert self.config.actor.load_weight == self.config.ref.load_weight
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                print(f'load ref weight start')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                load_megatron_model_weights(self.config,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            actor_model_config,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            ref_module,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            params_dtype=megatron_config.params_dtype,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            is_value_model=False)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After ref module init', logger=logger)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return ref_module, actor_model_config

        # TODO: add more optimizer args into config
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            optim_config = init_megatron_optim_config(optim_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            optim_config = None
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actor_optimizer = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return actor_module, hybrid_engine, actor_optimizer, actor_model_config, optim_config

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_rollout(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.rollout.name == 'vllm':
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.workers.sharding_manager import MegatronVLLMShardingManager
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.model import normalize_pp_vpp_params

            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_name_mapping = {
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "qkv_layer_name":
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.config.rollout.layer_name_map.get("qkv_layer_name", "qkv"),
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                "gate_proj_layer_name":
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.config.rollout.layer_name_map.get("gate_proj_layer_name", "linear_fc1.weight"),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            }

            # reshard the weight partition from actor to rollout to initialize the rollout class
            # create a new cuda space for parameters not in this pp rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.hybrid_engine.load_params_to_cuda()
            # broadcast the parameters from pp rank to other ranks
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.hybrid_engine.allgather_params()
            # obtain name to parameters in pp/vpp
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            params = self.hybrid_engine.get_all_params()
            # update the param name for the
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            params = normalize_pp_vpp_params(params=params,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             num_hidden_layers=self.actor_model_config.num_hidden_layers,
                                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                             layer_name='layers')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollout = vLLMRollout(actor_module=params,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  config=self.config.rollout,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  tokenizer=self.tokenizer,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  model_hf_config=self.actor_model_config,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  train_tp=mpu.get_tensor_model_parallel_world_size())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After building vllm rollout', logger=logger)

            # perform weight resharding between actor and rollout
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sharding_manager = MegatronVLLMShardingManager(module=self.hybrid_engine,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           inference_engine=rollout.inference_engine,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           model_config=self.actor_model_config,
                                                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                           layer_name_mapping=layer_name_mapping)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After building sharding manager', logger=logger)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            NotImplementedError('Only vllmRollout is supported with Megatron now')

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return rollout, sharding_manager

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import importlib
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            importlib.import_module(self.config.model.external_lib)

        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from omegaconf import OmegaConf
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.torch_dtypes import PrecisionType
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = torch.bfloat16

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron_config = OmegaConf.create({
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'sequence_parallel': self.config.actor.megatron.get('sequence_parallel', True),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'param_dtype': PrecisionType.to_str(torch_dtype),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        })

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron_config = init_model_parallel_config(megatron_config)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self._is_actor:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config = self.config.actor.optim
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config = None
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor_module, self.hybrid_engine, self.actor_optimizer, \
            self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model_path=self.config.model.path,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                megatron_config=megatron_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config=optim_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                override_model_config=override_model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.actor = MegatronPPOActor(config=self.config.actor,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          model_config=self.actor_model_config,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          megatron_config=megatron_config,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          actor_module=self.actor_module,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          actor_optimizer=self.actor_optimizer,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          actor_optimizer_config=self.actor_optim_config)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_rollout:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.rollout, self.sharding_manager = self._build_rollout()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_ref:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model_path=self.config.model.path,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                megatron_config=megatron_config,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                optim_config=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                override_model_config=override_model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.ref_policy = MegatronPPOActor(config=self.config.ref,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               model_config=self.ref_model_config,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               megatron_config=megatron_config,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               actor_module=self.ref_module,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               actor_optimizer=None,
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               actor_optimizer_config=None)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_actor(self, data: DataProto):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_actor

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.cuda()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before update policy', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dataloader = self.actor.make_minibatch_iterator(data=data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics = self.actor.update_policy(dataloader=dataloader)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = DataProto(meta_info={'metrics': metrics})
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output
    
    # @register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
    # def compute_log_prob(self, data: DataProto) -> DataProto:
    #     assert self._is_rollout
    #     output = self.actor.compute_log_prob(data=data)
    #     output = DataProto.from_dict(tensors={'old_log_probs': output})
    #     torch.cuda.empty_cache()
    #     return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def generate_sequences(self, prompts: DataProto):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_rollout

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompts.batch = prompts.batch.cuda()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompts.meta_info.update(meta_info)
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with self.sharding_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After entering sharding manager', logger=logger)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompts = self.sharding_manager.preprocess_data(prompts)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.rollout.generate_sequences(prompts=prompts)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            log_gpu_memory_usage('After rollout generation', logger=logger)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.sharding_manager.postprocess_data(output)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        validate = prompts.meta_info.get('validate', False)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_actor and not validate:
            # we should always recompute old_log_probs when it is HybridEngine
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.meta_info['temperature'] = self.config.rollout.temperature
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            old_log_probs = self.actor.compute_log_prob(data=output)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.batch['old_log_probs'] = old_log_probs

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # clear kv cache
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_ref_log_prob(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_ref
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_megatron_param_and_grad(self.ref_module, torch.cuda.current_device(), self._is_offload_grad)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_batch_size = self.config.rollout.log_prob_micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['micro_batch_size'] = micro_batch_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.meta_info['temperature'] = self.config.rollout.temperature
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = self.ref_policy.compute_log_prob(data=data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = DataProto.from_dict(tensors={'ref_log_prob': output})
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self._is_offload_param:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            offload_megatron_param_and_grad(self.ref_module, self._is_offload_grad)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_checkpoint(self, checkpoint_path):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_pretrained_model(self, checkpoint_path):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def save_checkpoint(self, checkpoint_path):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self._is_actor
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass


# 中文注释：下一行定义类，用于组织相关状态与行为。
class CriticWorker(MegatronWorker):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not torch.distributed.is_initialized():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank = int(os.environ['LOCAL_RANK'])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.init_process_group(backend="nccl")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_device(rank)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.megatron.sequence_parallel:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mpu.initialize_model_parallel(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                virtual_pipeline_model_parallel_size=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pipeline_model_parallel_split_rank=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_sharp=False,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                context_parallel_size=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                expert_model_parallel_size=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                nccl_communicator_config_path=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()

        # TODO(sgm): support critic model offload

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_critic_model_optimizer(self,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      model_path,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      megatron_config: ModelParallelConfig,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      optim_config,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      override_model_config,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      enable_gradient_checkpointing=False):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core.models.gpt.gpt_model import ModelType
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.model import print_model_size, update_model_config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config, init_model_parallel_config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = copy_local_path_from_hdfs(model_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the actor_model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_model_config = AutoConfig.from_pretrained(local_path)

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
        update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Model config after override: {critic_model_config}')

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def megatron_critic_model_provider(pre_process, post_process):
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.model import get_parallel_model_from_config
            # TODO: support vpp here
            # vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_model = get_parallel_model_from_config(config=critic_model_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            megatron_config=megatron_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            pre_process=pre_process,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            post_process=post_process,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            value=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_model.cuda()
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return parallel_model

        # Step 3: initialize the megatron model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_module = get_model(model_provider_func=megatron_critic_model_provider,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  model_type=ModelType.encoder_or_decoder,
                                  # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                  wrap_with_ddp=True)
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.load_weight:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_megatron_model_weights(self.config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        critic_model_config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        critic_module,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        params_dtype=megatron_config.params_dtype,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        is_value_model=True)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        optim_config = init_megatron_optim_config(optim_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return critic_module, critic_optimizer, critic_model_config, optim_config

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self):
        # create critic
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from omegaconf import OmegaConf
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.torch_dtypes import PrecisionType

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import importlib
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            importlib.import_module(self.config.model.external_lib)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = torch.bfloat16

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron_config = OmegaConf.create({
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'sequence_parallel': self.config.megatron.get('sequence_parallel', True),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'param_dtype': PrecisionType.to_str(torch_dtype),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        })

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron_config = init_model_parallel_config(megatron_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        critic_module, critic_optimizer, critic_model_config, critic_optimizer_config = self._build_critic_model_optimizer(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_path=self.config.model.path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            megatron_config=megatron_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            optim_config=self.config.optim,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            override_model_config=override_model_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.critic = MegatronPPOCritic(config=self.config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        model_config=critic_model_config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        megatron_config=megatron_config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        critic_module=critic_module,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        critic_optimizer=critic_optimizer,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        critic_optimizer_config=critic_optimizer_config)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_values(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        values = self.critic.compute_values(data=data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = DataProto.from_dict(tensors={'values': values})
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_critic(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data.to('cuda')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dataloader = self.critic.make_minibatch_iterator(data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        metrics = self.critic.update_critic(dataloader=dataloader)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = DataProto(batch=None, meta_info={'metrics': metrics})
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_checkpoint(self, checkpoint_path):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def save_checkpoint(self, checkpoint_path):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass


# 中文注释：下一行定义类，用于组织相关状态与行为。
class RewardModelWorker(MegatronWorker):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not torch.distributed.is_initialized():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank = int(os.environ['LOCAL_RANK'])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.distributed.init_process_group(backend="nccl")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_device(rank)

            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.config.megatron.sequence_parallel:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mpu.initialize_model_parallel(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                virtual_pipeline_model_parallel_size=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pipeline_model_parallel_split_rank=None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                use_sharp=False,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                context_parallel_size=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                expert_model_parallel_size=1,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                nccl_communicator_config_path=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config.micro_batch_size //= mpu.get_data_parallel_world_size()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_rm_model(self, model_path, megatron_config: ModelParallelConfig, override_model_config):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from megatron.core.models.gpt.gpt_model import ModelType
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.model import print_model_size, update_model_config
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.megatron_utils import get_model
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # Step 1: initialize the tokenizer
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_path = copy_local_path_from_hdfs(model_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = hf_tokenizer(local_path)

        # Step 2: get the actor_model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_model_config = AutoConfig.from_pretrained(local_path)

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
        update_model_config(rm_model_config, override_config_kwargs=override_config_kwargs)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.rank == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f'Model config after override: {rm_model_config}')

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def megatron_rm_model_provider(pre_process, post_process):
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from verl.utils.model import get_parallel_model_from_config
            # vpp is not supported yet because it will hang for some reason. Need debugging
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()  # this will be set inside get_model
            # this_megatron_config = copy.deepcopy(megatron_config)
            # this_megatron_config.virtual_pipeline_model_parallel_rank = vpp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_model = get_parallel_model_from_config(config=rm_model_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            megatron_config=megatron_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            pre_process=pre_process,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            post_process=post_process,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            value=True)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_model.cuda()
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return parallel_model

        # Step 3: initialize the megatron model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_model = get_model(model_provider_func=megatron_rm_model_provider,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 model_type=ModelType.encoder_or_decoder,
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 wrap_with_ddp=False)
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # reward_model = nn.ModuleList(reward_model)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.load_weight:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_megatron_model_weights(self.config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        rm_model_config,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        reward_model,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        params_dtype=megatron_config.params_dtype,
                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                        is_value_model=True)

        # TODO: add more optimizer args into config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return reward_model, rm_model_config

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self):
        # create critic
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from omegaconf import OmegaConf
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.torch_dtypes import PrecisionType
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from transformers import AutoTokenizer

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            import importlib
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            importlib.import_module(self.config.model.external_lib)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sft_tokenizer_local_path = copy_local_path_from_hdfs(self.config.model.input_tokenizer)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_tokenizer_path = self.config.model.get('rm_tokenizer', None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rm_tokenizer = None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if rm_tokenizer_path is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_tokenizer_local_path = copy_local_path_from_hdfs(rm_tokenizer_path)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rm_tokenizer = hf_tokenizer(rm_tokenizer_local_path)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch_dtype = torch.bfloat16

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron_config = OmegaConf.create({
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'sequence_parallel': self.config.megatron.get('sequence_parallel', True),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'param_dtype': PrecisionType.to_str(torch_dtype),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'tensor_model_parallel_size': mpu.get_tensor_model_parallel_world_size(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'pipeline_model_parallel_size': mpu.get_pipeline_model_parallel_world_size(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'virtual_pipeline_model_parallel_rank': mpu.get_virtual_pipeline_model_parallel_rank(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'virtual_pipeline_model_parallel_size': mpu.get_virtual_pipeline_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        })

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        megatron_config = init_model_parallel_config(megatron_config)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        reward_model_module, reward_model_config = self._build_rm_model(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_path=self.config.model.path,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            megatron_config=megatron_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            override_model_config=override_model_config,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rm = MegatronRewardModel(config=self.config,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      reward_model_module=reward_model_module,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      model_config=reward_model_config,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      megatron_config=megatron_config,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      sft_tokenizer=sft_tokenizer,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      rm_tokenizer=rm_tokenizer)

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def compute_rm_score(self, data: DataProto):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = data.batch.cuda()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = self.rm.compute_reward(data)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = output.to('cpu')
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output
