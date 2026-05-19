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
import logging
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed.device_mesh import DeviceMesh

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.third_party.vllm import LLM
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.third_party.vllm import parallel_state as vllm_ps
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.debug import log_gpu_memory_usage

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .base import BaseShardingManager

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = logging.getLogger(__file__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


# 中文注释：下一行定义类，用于组织相关状态与行为。
class FSDPVLLMShardingManager(BaseShardingManager):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 module: FSDP,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 inference_engine: LLM,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 model_config,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 full_params: bool = False,
                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                 device_mesh: DeviceMesh = None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module = module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine = inference_engine
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_mesh = device_mesh

        # Full params
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.full_params = full_params
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if full_params:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            FSDP.set_state_dict_type(self.module,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     state_dict_config=FullStateDictConfig())
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            FSDP.set_state_dict_type(self.module,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.gen_random_states = torch.cuda.get_rng_state()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_rng_state(self.torch_random_states)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.gen_random_states = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __enter__(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params = self.module.state_dict()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_format = 'hf' if self.full_params else 'dtensor'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        del params
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.torch_random_states = torch.cuda.get_rng_state()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_rng_state(self.gen_random_states)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __exit__(self, exc_type, exc_value, traceback):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine.offload_model_weights()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.train()

        # add empty cache after each compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

        # restore random states
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_mesh is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.gen_random_states = torch.cuda.get_rng_state()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_rng_state(self.torch_random_states)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            group=vllm_ps.get_tensor_model_parallel_group(),
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            dim=0)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        broadcast_dict_tensor(data.batch,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              src=vllm_ps.get_tensor_model_parallel_src_rank(),
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              group=vllm_ps.get_tensor_model_parallel_group())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp_rank = torch.distributed.get_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_prompts = data.chunk(chunks=tp_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = local_prompts[dp_rank % tp_size]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data
