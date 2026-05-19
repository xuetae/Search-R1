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
from typing import Dict, List, Tuple, Optional, Union, Set

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor import InputMetadata, set_random_seed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.parallel_utils.parallel_state import (initialize_model_parallel)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sampling_params import SamplingParams, SamplingType
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.worker.cache_engine import CacheEngine
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .model_runner import ModelRunner
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .model_loader import load_weights
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .parallel_state import initialize_model_parallel_from_megatron
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest


# 中文注释：下一行定义类，用于组织相关状态与行为。
class Worker:
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
        rank: Optional[int] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        distributed_init_method: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kv_cache_dtype: Optional[str] = "auto",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # self.model = model  # will be replaced in the init_model
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parallel_config = parallel_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler_config = scheduler_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.rank = rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.distributed_init_method = distributed_init_method
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_config = lora_config

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_runner = ModelRunner(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=self.lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_cache_dtype=kv_cache_dtype,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.block_size = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.sliding_window = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_engine = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_events = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gpu_cache = None

        # For offloading inference engine params
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cpu_model = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_model(self, cupy_port: Optional[int] = None):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # Env vars will be set by TORCHRUN.
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

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _check_if_gpu_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        # TODO: do not use cupy
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        _init_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.parallel_config.disable_custom_all_reduce:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            init_custom_ar()
        # Initialize the model.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_random_seed(self.model_config.seed)
        # self.model = get_model(actor_model=self.model, model_config=self.model_config)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_model(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_runner.load_model()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.inference_mode()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def profile_num_available_blocks(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        block_size: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpu_memory_utilization: float,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cpu_swap_space: int,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_dtype: str,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> Tuple[int, int]:
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

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_block_size = CacheEngine.get_cache_block_size(block_size, cache_dtype, self.model_config,
                                                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                            self.parallel_config)
        # NOTE(sgm) use the remaining memory
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = int((free_gpu_memory * gpu_memory_utilization) // cache_block_size)
        # num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = max(num_gpu_blocks, 0)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = max(num_cpu_blocks, 0)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.model_runner.lora_manager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_runner.remove_all_loras()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gc.collect()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()
        # Synchronize number of blocks with all the rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = torch.tensor([num_gpu_blocks], device='cuda')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = torch.tensor([num_cpu_blocks], device='cuda')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.all_reduce(num_gpu_blocks,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     op=torch.distributed.ReduceOp.MIN,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     group=get_tensor_model_parallel_group())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.all_reduce(num_cpu_blocks,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     op=torch.distributed.ReduceOp.MIN,
                                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                     group=get_tensor_model_parallel_group())
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = num_gpu_blocks.item()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = num_cpu_blocks.item()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return num_gpu_blocks, num_cpu_blocks

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.cache_engine is None and self.gpu_cache is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_config = cache_config
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_engine = CacheEngine(self.cache_config, self.model_config, self.parallel_config)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_events = self.cache_engine.events
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.gpu_cache = self.cache_engine.gpu_cache
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_runner.set_block_size(self.cache_engine.block_size)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def free_cache_engine(self):
        # ensure `enforce_eager=True`
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_engine = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.gpu_cache = None

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def warm_up_model(self) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not self.model_config.enforce_eager:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_random_seed(self.model_config.seed)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def cache_swap(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        blocks_to_swap_in: Dict[int, int],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        blocks_to_swap_out: Dict[int, int],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        blocks_to_copy: Dict[int, List[int]],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # Issue cache operations.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        issued_cache_op = False
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if blocks_to_swap_in:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_engine.swap_in(blocks_to_swap_in)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            issued_cache_op = True
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if blocks_to_swap_out:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_engine.swap_out(blocks_to_swap_out)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            issued_cache_op = True
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if blocks_to_copy:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_engine.copy(blocks_to_copy)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            issued_cache_op = True

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if cache_events is not None:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for event in cache_events:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                event.wait()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @torch.inference_mode()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_model(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list: List[SequenceGroupMetadata],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        blocks_to_swap_in: Dict[int, int],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        blocks_to_swap_out: Dict[int, int],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        blocks_to_copy: Dict[int, List[int]],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> SamplerOutput:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_seq_groups = len(seq_group_metadata_list)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num_seq_groups == 0:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = self.model_runner.execute_model(seq_group_metadata_list, self.gpu_cache)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

        # # Prepare input tensors.
        # # NOTE(shengguangming): currently we pad in our dataloader and unpad it in pre_process_input, j
        # # we can just input un-padded sequence for better performance
        # input_tokens, input_positions, input_metadata = self._prepare_inputs(seq_group_metadata_list)

        # # Execute the model.
        # output = self.model(
        #     input_ids=input_tokens,
        #     positions=input_positions,
        #     kv_caches=self.gpu_cache,
        #     input_metadata=input_metadata,
        #     cache_events=cache_events,
        # )
        # return output

    # assume the input is .state_dict()
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def sync_model_weights(self, actor_weights: Dict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_weights(actor_weights, self.model_runner.model)

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
    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.model_runner.add_lora(lora_request)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def remove_lora(self, lora_id: int) -> bool:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.model_runner.remove_lora(lora_id)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def list_loras(self) -> Set[int]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.model_runner.list_loras()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _init_distributed_environment(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parallel_config: ParallelConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    rank: int,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    distributed_init_method: Optional[str] = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> None:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Initialize the distributed environment."""
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.is_initialized():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print('The distributed environment has been initialized before vLLM')
    # 中文注释：下一行继续判断其他条件分支。
    elif not distributed_init_method:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise ValueError("distributed_init_method must be set if torch.distributed "
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         "is not already initialized")
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.distributed.init_process_group(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            backend="nccl",
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            world_size=parallel_config.world_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank=rank,
            # init_method=distributed_init_method,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

    # A small all_reduce for warmup.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    # TODO (shengguangming): maybe we should also flag the megatron is initialized
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch.distributed.get_world_size() > 1:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        initialize_model_parallel_from_megatron(tensor_model_parallel_size=parallel_config.tensor_parallel_size)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        initialize_model_parallel()


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _pad_to_alignment(x: List[int], multiple_of: int, pad: int) -> List[int]:
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x + [pad] * ((-len(x)) % multiple_of)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return x + [pad] * (max_len - len(x))


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if torch_dtype == torch.bfloat16:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        compute_capability = torch.cuda.get_device_capability()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if compute_capability[0] < 8:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpu_name = torch.cuda.get_device_name()
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("Bfloat16 is only supported on GPUs with compute capability "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"{compute_capability[0]}.{compute_capability[1]}.")
