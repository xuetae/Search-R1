# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""A GPU worker class."""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import gc
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, List, Tuple, Optional, Union, Type

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         SchedulerConfig, SpeculativeConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor import set_random_seed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors, SamplerOutput)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.cache_engine import CacheEngine
# TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.distributed import (init_distributed_environment, set_custom_all_reduce, get_tensor_model_parallel_group)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.worker_base import WorkerInput
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.model_runner import GPUModelRunnerBase
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .model_runner import ModelRunner
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .megatron_weight_loaders import load_megatron_weights
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .hf_weight_loader import load_hf_weights
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .dtensor_weight_loaders import load_dtensor_weights
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .parallel_state import (ensure_model_parallel_initialized)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import ModelConfig, LoadConfig, LoadFormat


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Worker(Worker):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_config: LoadConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_rank: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        distributed_init_method: str,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        multimodal_config: Optional[MultiModalConfig] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        speculative_config: Optional[SpeculativeConfig] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_driver_worker: bool = False,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # self.model = model  # will be replaced in the init_model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parallel_config = parallel_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parallel_config.rank = rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler_config = scheduler_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_config = device_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config = cache_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.local_rank = local_rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rank = rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.distributed_init_method = distributed_init_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_config = lora_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.load_config = load_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.prompt_adapter_config = prompt_adapter_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.is_driver_worker = is_driver_worker  # TODO: we don't need driver
        # if parallel_config and is_driver_worker:
        #     assert rank % parallel_config.tensor_parallel_size == 0, \
        #            "Driver worker should be rank 0 of tensor parallel group."
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
            from vllm.utils import init_cached_hf_modules
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            init_cached_hf_modules()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.multimodal_config = multimodal_config

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                not in ["medusa", "mlp_speculator"]) \
                    else {"return_hidden_states": True}

        # TODO(sgm): set correct model runner class
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_runner_cls is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ModelRunnerClass = model_runner_cls
        # 中文注释：下一行继续判断其他条件分支。
        elif self.model_config.embedding_mode:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ModelRunnerClass = EmbeddingModelRunner
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model, # [VERL]: add for verl
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_config=load_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=self.lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_cache_dtype=self.cache_config.cache_dtype,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            is_driver_worker=is_driver_worker,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_adapter_config=prompt_adapter_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            multimodal_config=multimodal_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            **speculative_args,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_engine: List[CacheEngine] = None
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None

        # NOTE(sgm): [VERL] For offloading inference engine params
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cpu_model = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_device(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.rank = self.rank if self.rank is not None else int(os.getenv("RANK", "-1"))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.device = torch.device(f"cuda:{local_rank}")
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self.rank < 0:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError("Invalid or unspecified rank.")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.set_device(self.device)

            # Use the world_size set by TORCHRUN
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            world_size = int(os.getenv("WORLD_SIZE", "-1"))
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.parallel_config.world_size = world_size

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _check_if_gpu_supports_dtype(self.model_config.dtype)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.cuda.empty_cache()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize the distributed environment.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_worker_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method,
                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                            self.local_rank)
        # Set random seed.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_random_seed(self.model_config.seed)
        # self.model = get_model(actor_model=self.model, model_config=self.model_config)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.inference_mode()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.synchronize()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        peak_memory = total_gpu_memory - free_gpu_memory

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert peak_memory > 0, ("Error in memory profiling. This happens when the GPU memory was "
                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                 "not properly cleaned up before initializing the vLLM instance.")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_block_size = self.get_cache_block_size_bytes()

        # NOTE(sgm) [VERL] use the remaining memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = int((free_gpu_memory * self.cache_config.gpu_memory_utilization) // cache_block_size)
        # num_gpu_blocks = int((total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_memory) // cache_block_size)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = max(num_gpu_blocks, 0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = max(num_cpu_blocks, 0)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.model_runner.lora_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_runner.remove_all_loras()

        # NOTE(sgm): Add for [VERL], synchronize number of blocks with all the rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = torch.tensor([num_gpu_blocks], device='cuda')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = torch.tensor([num_cpu_blocks], device='cuda')

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.all_reduce(num_gpu_blocks,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     op=torch.distributed.ReduceOp.MIN,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     group=get_tensor_model_parallel_group().device_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.all_reduce(num_cpu_blocks,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     op=torch.distributed.ReduceOp.MIN,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     group=get_tensor_model_parallel_group().device_group)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = num_gpu_blocks.item()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = num_cpu_blocks.item()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gc.collect()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return num_gpu_blocks, num_cpu_blocks

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_cache_engine(self):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.cache_engine is None and self.gpu_cache is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            super()._init_cache_engine()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def free_cache_engine(self):
        # ensure `enforce_eager=True`
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_engine = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gpu_cache = None

    # NOTE(sgm): [VERL]: adapt from _execute_model_spmd()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_model(self,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      execute_model_req: ExecuteModelRequest,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      intermediate_tensors: Optional[IntermediateTensors] = None) -> Optional[List[SamplerOutput]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert execute_model_req is not None, ("_execute_model_spmd() requires each worker to take in an "
                                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                               "ExecuteModelRequest")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        worker_input: WorkerInput = self.prepare_worker_input(execute_model_req=execute_model_req)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_input: ModelRunnerInputBase = (self.model_runner.prepare_model_input(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            execute_model_req.seq_group_metadata_list))

        # verl.worker.workerbase.WorkerBase
        # swap cache
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if worker_input.num_seq_groups == 0:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return []

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.model_runner.execute_model(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_input, self.kv_cache[worker_input.virtual_engine] if self.kv_cache is not None else None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            intermediate_tensors)

    # assume the input is .state_dict()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def sync_model_weights(self, actor_weights: Dict, load_format: str):
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if load_format in [LoadFormat.MEGATRON, LoadFormat.AUTO]:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_megatron_weights(actor_weights, self.model_runner.model)
        # 中文注释：下一行继续判断其他条件分支。
        elif load_format == LoadFormat.HF:
            # full model state dict without no sharding
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_hf_weights(actor_weights, self.model_runner.model)
        # 中文注释：下一行继续判断其他条件分支。
        elif load_format == LoadFormat.DTENSOR:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            load_dtensor_weights(actor_weights, self.model_runner.model)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def offload_model_weights(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.cpu_model == None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cpu_model = {}
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for name, params in self.model_runner.model.named_parameters():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.cpu_model[name] = torch.empty_like(params, device='cpu')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                params.data = self.cpu_model[name]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for name, params in self.model_runner.model.named_parameters():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                params.data = self.cpu_model[name]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def init_worker_distributed_environment(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parallel_config: ParallelConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    distributed_init_method: Optional[str] = "env://",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_rank: int = -1,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Initialize the distributed environment."""
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # NOTE(sgm) use tcp://localhost:xxxx will hang in HF setting without megatron
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ensure_model_parallel_initialized(tensor_model_parallel_size=parallel_config.tensor_parallel_size,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      pipeline_model_parallel_size=parallel_config.pipeline_parallel_size)

    # TODO(sgm): check whether need this
    # if pynccl_utils.is_initialized():
    #     pynccl_world_size = pynccl_utils.get_world_size()
    #     if pynccl_world_size != parallel_config.world_size:
    #         raise RuntimeError(
    #             "pynccl is already initialized but the pynccl world "
    #             "size does not match parallel_config.world_size "
    #             f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    # elif parallel_config.world_size > 1:
    #     # NOTE(woosuk): We don't initialize pynccl process group when world size
    #     # is 1.
    #     # NOTE(kaichao): By default, pynccl is initialized for tp group.
    #     pynccl_utils.init_process_group(
    #         group=get_tensor_model_parallel_cpu_group())

    # # Initialize a custom fast all-reduce implementation.
    # if not parallel_config.disable_custom_all_reduce:
    #     init_custom_ar()

    # A small all_reduce for warmup.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # if pynccl_utils.is_initialized():
    #     pynccl_utils.all_reduce(torch.zeros(1).cuda())
