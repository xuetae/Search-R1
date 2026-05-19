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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import socket
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Any, Dict, List, Optional, Set, Tuple

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import vllm.envs as envs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.executor.executor_base import ExecutorBase, ExecutorAsyncBase
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sequence import SamplerOutput, ExecuteModelRequest

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                         SchedulerConfig, SpeculativeConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .config import ModelConfig, LoadConfig

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class SPMDGPUExecutor(ExecutorBase):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """SPMD-based multi-GPU executor implementations."""

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model, # pytorch model itself or its parameter dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_config: ModelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cache_config: CacheConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config: ParallelConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduler_config: SchedulerConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        device_config: DeviceConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        load_config: LoadConfig,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        multimodal_config: Optional[MultiModalConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        speculative_config: Optional[SpeculativeConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_adapter_config: Optional[PromptAdapterConfig],
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config = cache_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_config = lora_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.load_config = load_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parallel_config = parallel_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler_config = scheduler_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_config = device_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.multimodal_config = multimodal_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.speculative_config = speculative_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.prompt_adapter_config = prompt_adapter_config

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        distributed_init_method = initialize_cluster(parallel_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_executor(model, distributed_init_method)

    # TODO(sgm): verl not support speculative decode now
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_executor(self, model, distributed_init_method) -> None:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert (not self.speculative_config), "Speculative decoding not yet supported for multi-GPU backend."

        # Create the parallel worker for each GPU.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_workers_sp(model, distributed_init_method)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_workers_sp(self, model, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from .worker import Worker  # pylint: disable=import-outside-toplevel

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = int(os.getenv("RANK"))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_rank = int(os.getenv("LOCAL_RANK"))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'local rank {local_rank}')

        # see https://github.com/NVIDIA/nccl/issues/1234
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker = Worker(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.model_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.parallel_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.scheduler_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.device_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.cache_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.load_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            distributed_init_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=self.lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            multimodal_config=self.multimodal_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            speculative_config=None,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_adapter_config=self.speculative_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            is_driver_worker=True,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model_runner_cls=None,  # use the default one
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # NOTE(shengguangming): torch.distributed.init_process_group will be called inside the init_model()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.init_device()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.load_model()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Determine the number of available KV blocks.

        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.

        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_blocks = self.worker.determine_num_available_blocks()

        # NOTE(shengguangming): Now we don't use a shared centralized controler but each process will
        # have its own scheduler
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = num_blocks[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = num_blocks[1]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return num_gpu_blocks, num_cpu_blocks

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Initialize the KV cache in all workers.
        """

        # NOTE: We log here to avoid multiple logs when number of workers is
        # greater than one. We could log in the engine, but not all executors
        # have GPUs.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks, num_cpu_blocks)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'before init cache memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.initialize_cache(num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=num_cpu_blocks)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if torch.distributed.get_rank() == 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'after init cache memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB'
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

    # NOTE(sgm): This will not profile & capture the model(CUDAGraph) when rebuilding KVCache
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_cache_engine(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker._init_cache_engine()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def free_cache_engine(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.free_cache_engine()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_model(self, execute_model_req) -> List[SamplerOutput]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_outputs = self.worker.execute_model(execute_model_req=execute_model_req)

        # NOTE(sgm):
        # Each GPU in vllm under verl has its own spmd_gpu_executor, therefore all GPUs should return the outputs
        # In vllm with ray, only the driver worker returns the sampling results.
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return all_outputs

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.add_lora(lora_request=lora_request)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def remove_lora(self, lora_id: int) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert lora_id > 0, "lora_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.remove_lora(lora_id=lora_id)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def list_loras(self) -> Set[int]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.list_loras()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def check_health(self) -> None:
        # SPMDExecutor will always be healthy as long as
        # it's running.
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return

    # NOTE(sgm) add for verl to pass the abstract class test, not used
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from vllm.prompt_adapter.request import PromptAdapterRequest

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_prompt_adapter(self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert prompt_adapter_request.prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.add_prompt_adapter(prompt_adapter_request)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def list_prompt_adapters(self) -> Set[int]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.list_prompt_adapters()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pin_lora(self, lora_id: int) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert lora_id > 0, "lora_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.pin_lora(lora_id)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert prompt_adapter_id > 0, \
                "prompt_adapter_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.pin_prompt_adapter(prompt_adapter_id)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert prompt_adapter_id > 0, \
            "prompt_adapter_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.remove_prompt_adapter(prompt_adapter_id)

    # NOTE(sgm): add for verl
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def offload_model_weights(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.offload_model_weights()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.sync_model_weights(actor_weights=actor_weights, load_format=load_format)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def initialize_cluster(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    parallel_config: ParallelConfig,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    engine_use_ray: bool = False,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ray_address: Optional[str] = None,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
) -> Tuple[str, Optional[None]]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.

    Returns:
        The `distributed_init_method` is the address for initializing the
        distributed backend.
    """

    # Initialize cluster locally.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    port = get_open_port()
    # We need to setup the distributed init method to make sure
    # the distributed megatron code (e.g., get world size) works correctly.
    # distributed_init_method = f"tcp://localhost:{port}"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    distributed_init_method = 'env://'
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return distributed_init_method


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_open_port():
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        s.bind(("", 0))
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return s.getsockname()[1]


# TODO(sgm): not implemented async executor yet
# 中文注释：下一行定义类，用于组织相关状态与行为。
class SPMDGPUExecutorAsync(SPMDGPUExecutor, ExecutorAsyncBase):

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Executes one model step on the given sequences."""
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplementedError

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    async def check_health_async(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.check_health()
