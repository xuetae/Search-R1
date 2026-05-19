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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import socket
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import time
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.lora.request import LoRARequest
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.core.scheduler import Scheduler, SchedulerOutputs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.logger import init_logger
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.outputs import RequestOutput
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sampling_params import SamplingParams
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           SequenceOutput, SequenceStatus)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.transformers_utils.tokenizer import detokenize_incrementally
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.engine.metrics import StatLogger, Stats
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from vllm.utils import Counter
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.nn as nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .arg_utils import EngineArgs
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .tokenizer import TokenizerGroup

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
logger = init_logger(__name__)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_LOCAL_LOGGING_INTERVAL_SEC = 5


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LLMEngine:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer: nn.Module,
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
        lora_config: Optional[LoRAConfig],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        distributed_init_method: str,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        placement_group: Optional[None],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        log_stats: bool,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.info("Initializing an LLM engine with config: "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"model={model_config.model!r}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"tokenizer={model_config.tokenizer!r}, "
                    # f"tokenizer_mode={model_config.tokenizer_mode}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"revision={model_config.revision}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"tokenizer_revision={model_config.tokenizer_revision}, "
                    # f"trust_remote_code={model_config.trust_remote_code}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"dtype={model_config.dtype}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"max_seq_len={model_config.max_model_len}, "
                    # f"download_dir={model_config.download_dir!r}, "
                    # f"load_format={model_config.load_format}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"disable_custom_all_reduce={parallel_config.disable_custom_all_reduce}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"quantization={model_config.quantization}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config  # TODO: currently is hfconfig
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config = cache_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.lora_config = lora_config
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.cache_config.sliding_window == getattr(self.model_config.hf_config, "sliding_window", None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.parallel_config = parallel_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler_config = scheduler_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.device_config = device_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.log_stats = log_stats
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._verify_args()

        # self.model = model # should not store the model, it should be deleted
        # TODO(shengguangming): maybe we can choose init here or from arguments
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_tokenizer(tokenizer)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.seq_counter = Counter()

        # Create the parallel GPU workers.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_workers_sp(model, distributed_init_method)

        # Profile the memory usage and initialize the cache.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._init_cache_sp()

        # Create the scheduler.
        # NOTE(shengguangming): each process will have independent scheduler
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # Metric Logging.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.log_stats:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.stat_logger = StatLogger(local_interval=_LOCAL_LOGGING_INTERVAL_SEC)

        # Logging.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.num_generation_tokens: List[Tuple[float, int]] = []

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_tokenizer(self, tokenizer, **tokenizer_init_kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_kwargs = dict(enable_lora=bool(self.lora_config),
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           max_num_seqs=self.scheduler_config.max_num_seqs,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           max_input_length=None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        init_kwargs.update(tokenizer_init_kwargs)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer: TokenizerGroup = TokenizerGroup(tokenizer, **init_kwargs)

    # TODO: check get_lora_tokenizer func
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_tokenizer_for_seq(self, sequence: Sequence):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.tokenizer.get_lora_tokenizer(sequence.lora_request)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_workers_sp(self, model, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from .worker import Worker  # pylint: disable=import-outside-toplevel

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = int(os.getenv("RANK"))

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
            rank,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            distributed_init_method,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            lora_config=self.lora_config,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_cache_dtype=self.cache_config.cache_dtype,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # NOTE(shengguangming): torch.distributed.init_process_group will be called inside the init_model()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.init_model()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.load_model()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _verify_args(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config.verify_with_parallel_config(self.parallel_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _init_cache_sp(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_blocks = self.worker.profile_num_available_blocks(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            block_size=self.cache_config.block_size,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_swap_space=self.cache_config.swap_space_bytes,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cache_dtype=self.cache_config.cache_dtype,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # NOTE(shengguangming): Now we don't use a shared centralized controler but each process will
        # have its own scheduler
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpu_blocks = num_blocks[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_cpu_blocks = num_blocks[1]

        # FIXME(woosuk): Change to debug log.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    f"# CPU blocks: {num_cpu_blocks}")

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num_gpu_blocks <= 0:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError("No available memory for the cache blocks. "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "Try increasing `gpu_memory_utilization` when "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "initializing the engine.")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.model_config.max_model_len > max_seq_len:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"The model's max seq len ({self.model_config.max_model_len}) "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "is larger than the maximum number of tokens that can be "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             f"stored in KV cache ({max_seq_len}). Try increasing "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "`gpu_memory_utilization` or decreasing `max_model_len` when "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "initializing the engine.")

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.init_cache_engine(cache_config=self.cache_config)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.warm_up_model()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def init_cache_engine(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.init_cache_engine(cache_config=self.cache_config)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def free_cache_engine(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.free_cache_engine()

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def from_engine_args(cls, model, tokenizer, engine_args: EngineArgs) -> "LLMEngine":
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        engine_configs = engine_args.create_engine_configs()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        distributed_init_method, placement_group = initialize_cluster(parallel_config)
        # Create the LLM engine.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        engine = cls(model,
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     tokenizer,
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     *engine_configs,
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     distributed_init_method,
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     placement_group,
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     log_stats=not engine_args.disable_log_stats)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return engine

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_request(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        request_id: str,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt: Optional[str],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sampling_params: SamplingParams,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_token_ids: Optional[List[int]] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        arrival_time: Optional[float] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        lora_request: Optional[LoRARequest] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prefix_pos: Optional[int] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.

        Details:
            - Set arrival_time to the current time if it is None.
            - Set prompt_token_ids to the encoded prompt if it is None.
            - Create `best_of` number of :class:`~vllm.Sequence` objects.
            - Create a :class:`~vllm.SequenceGroup` object
              from the list of :class:`~vllm.Sequence`.
            - Add the :class:`~vllm.SequenceGroup` object to the scheduler.

        Example:
            >>> # initialize engine
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> # set request arguments
            >>> example_prompt = "Who is the president of the United States?"
            >>> sampling_params = SamplingParams(temperature=0.0)
            >>> request_id = 0
            >>>
            >>> # add the request to the engine
            >>> engine.add_request(
            >>>    str(request_id),
            >>>    example_prompt,
            >>>    SamplingParams(temperature=0.0))
            >>> # continue the request processing
            >>> ...
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if lora_request is not None and not self.lora_config:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                             "not enabled!")
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if arrival_time is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            arrival_time = time.monotonic()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if prompt_token_ids is None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert prompt is not None
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        block_size = self.cache_config.block_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_id = next(self.seq_counter)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, lora_request)

        # Check whether the input specifies prefix
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prefix = self.scheduler.prefix_pool.add_or_get_prefix(prompt_token_ids[:prefix_pos], lora_request.lora_int_id if
                                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                              lora_request else 0) if prefix_pos is not None else None

        # Create the sequence group.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group = SequenceGroup(request_id, [seq], sampling_params, arrival_time, lora_request, prefix)

        # Add the sequence group to the scheduler.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler.add_seq_group(seq_group)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.

        Details:
            - Refer to the
              :meth:`~vllm.core.scheduler.Scheduler.abort_seq_group`
              from class :class:`~vllm.core.scheduler.Scheduler`.

        Example:
            >>> # initialize engine and add a request with request_id
            >>> request_id = str(0)
            >>> # abort the request
            >>> engine.abort_request(request_id)
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler.abort_seq_group(request_id)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_model_config(self) -> ModelConfig:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Gets the model configuration."""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.model_config

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_num_unfinished_requests(self) -> int:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Gets the number of unfinished requests."""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.scheduler.get_num_unfinished_seq_groups()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def has_unfinished_requests(self) -> bool:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Returns True if there are unfinished requests."""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.scheduler.has_unfinished_seqs()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_beam_search_early_stopping(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        early_stopping: Union[bool, str],
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sampling_params: SamplingParams,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        best_running_seq: Sequence,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        current_worst_seq: Sequence,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert sampling_params.use_beam_search
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        length_penalty = sampling_params.length_penalty
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if early_stopping is True:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return True

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        current_worst_score = (current_worst_seq.get_beam_search_score(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(current_worst_seq).eos_token_id))
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if early_stopping is False:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert early_stopping == "never"
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                max_possible_length = max(best_running_seq.get_prompt_len() + sampling_params.max_tokens,
                                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                          self.scheduler_config.max_model_len)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                highest_attainable_score = (best_running_seq.get_beam_search_score(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    length_penalty=length_penalty,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    seq_len=max_possible_length))
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                highest_attainable_score = (best_running_seq.get_beam_search_score(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    length_penalty=length_penalty,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    eos_token_id=self.get_tokenizer_for_seq(best_running_seq).eos_token_id))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _process_sequence_group_outputs(self, seq_group: SequenceGroup, outputs: SequenceGroupOutput) -> None:

        # Process prompt logprobs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        prompt_logprobs = outputs.prompt_logprobs
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if prompt_logprobs is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        samples = outputs.samples
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        existing_finished_seqs = seq_group.get_finished_seqs()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        parent_child_dict = {parent_seq.seq_id: [] for parent_seq in parent_seqs}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for sample in samples:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for parent in parent_seqs:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            child_samples: List[SequenceOutput] = parent_child_dict[parent.seq_id]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                parent.status = SequenceStatus.FINISHED_ABORTED
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_group.remove(parent.seq_id)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler.free_seq(parent)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # Fork the parent sequence if there are multiple child samples.
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for child_sample in child_samples[:-1]:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                new_child_seq_id = next(self.seq_counter)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                child = parent.fork(new_child_seq_id)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                child.append_token_id(child_sample.output_token, child_sample.logprobs)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            last_child_sample = child_samples[-1]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            parent.append_token_id(last_child_sample.output_token, last_child_sample.logprobs)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            child_seqs.append((parent, parent))

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq, _ in child_seqs:
            # self._decode_sequence(seq, seq_group.sampling_params)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for seq, parent in child_seqs:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if seq is not parent:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    seq_group.add(seq)
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if not seq.is_finished():
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for seq, parent in child_seqs:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if seq is parent and seq.is_finished():
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.scheduler.free_seq(seq)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        selected_child_seqs = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        unselected_child_seqs = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        beam_width = seq_group.sampling_params.best_of
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        existing_finished_seqs = [(seq, None, False) for seq in existing_finished_seqs]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs if seq.is_finished()]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id),
                               # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                               reverse=True)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                selected_child_seqs.append((seq, parent))
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                unselected_child_seqs.append((seq, parent))
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs if not seq.is_finished()]
        # Sort the running sequences by their scores.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            length_penalty=length_penalty, eos_token_id=self.get_tokenizer_for_seq(x[0]).eos_token_id),
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                reverse=True)

        # Check if we can stop the beam search.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            stop_beam_search = True
        # 中文注释：下一行继续判断其他条件分支。
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            stop_beam_search = False
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # Check the early stopping criteria
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            best_running_seq = running_child_seqs[0][0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            stop_beam_search = self._check_beam_search_early_stopping(seq_group.sampling_params.early_stopping,
                                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                      seq_group.sampling_params, best_running_seq,
                                                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                                      current_worst_seq)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            unselected_child_seqs.extend(running_child_seqs)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq, parent in selected_child_seqs:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if seq is not parent:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_group.add(seq)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if not seq.is_finished():
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq, parent in selected_child_seqs:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if seq is parent and seq.is_finished():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq, parent in unselected_child_seqs:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_group.remove(seq.seq_id)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.scheduler.free_seq(seq)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _process_model_outputs(self, output: SamplerOutput, scheduler_outputs: SchedulerOutputs) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq_group, outputs in zip(scheduled_seq_groups, output):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._process_sequence_group_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        request_outputs: List[RequestOutput] = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq_group in scheduled_seq_groups:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            request_output = RequestOutput.from_seq_group(seq_group)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            request_outputs.append(request_output)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq_group in scheduler_outputs.ignored_seq_groups:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            request_output = RequestOutput.from_seq_group(seq_group)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            request_outputs.append(request_output)

        # Update prefix state, now all the uncomputed prefixes are computed.
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for seq_group in scheduled_seq_groups:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if (seq_group.prefix is not None and seq_group.prefix.allocated and not seq_group.prefix.computed):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq_group.prefix.computed = True

        # Log stats.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.log_stats:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.stat_logger.log(self._get_stats(scheduler_outputs))

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return request_outputs

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def step(self) -> List[RequestOutput]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not scheduler_outputs.is_empty():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.worker.execute_model(
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        seq_group_metadata_list=seq_group_metadata_list, # TODO: check this input
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        blocks_to_copy=scheduler_outputs.blocks_to_copy,)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return [RequestOutput.from_seq_group(seq_group) for seq_group in scheduler_outputs.ignored_seq_groups]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._process_model_outputs(output, scheduler_outputs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def do_log_stats(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Forced log when no requests active."""
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.log_stats:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.stat_logger.log(self._get_stats(scheduler_outputs=None))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _get_stats(self, scheduler_outputs: Optional[SchedulerOutputs]) -> Stats:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Get Stats to be Logged to Prometheus."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        now = time.monotonic()

        # KV Cache Usage in %.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_total_gpu = self.cache_config.num_gpu_blocks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_free_gpu = self.scheduler.block_manager.get_num_free_gpu_blocks()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        gpu_cache_usage = 1.0 - (num_free_gpu / num_total_gpu)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_total_cpu = self.cache_config.num_cpu_blocks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cpu_cache_usage = 0.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num_total_cpu > 0:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_free_cpu = self.scheduler.block_manager.get_num_free_cpu_blocks()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_cache_usage = 1.0 - (num_free_cpu / num_total_cpu)

        # Scheduler State
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_running = len(self.scheduler.running)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_swapped = len(self.scheduler.swapped)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_waiting = len(self.scheduler.waiting)

        # Iteration stats if we have scheduler output.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_prompt_tokens = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_generation_tokens = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        time_to_first_tokens = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        time_per_output_tokens = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        time_e2e_requests = []
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if scheduler_outputs is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prompt_run = scheduler_outputs.prompt_run

            # Number of Tokens.
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if prompt_run:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_prompt_tokens = scheduler_outputs.num_batched_tokens
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                num_generation_tokens = scheduler_outputs.num_batched_tokens

            # Latency Timings.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time_last_iters = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                # Time since last token. (n.b. updates seq_group.last_token_time)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                time_last_iters.append(seq_group.get_last_latency(now))
                # Time since arrival for all finished requests.
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if seq_group.is_finished():
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    time_e2e_requests.append(now - seq_group.arrival_time)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time_to_first_tokens = time_last_iters if prompt_run else []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time_per_output_tokens = [] if prompt_run else time_last_iters

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return Stats(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            now=now,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_running=num_running,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_swapped=num_swapped,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_waiting=num_waiting,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gpu_cache_usage=gpu_cache_usage,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cpu_cache_usage=cpu_cache_usage,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_prompt_tokens=num_prompt_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_generation_tokens=num_generation_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time_to_first_tokens=time_to_first_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time_per_output_tokens=time_per_output_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            time_e2e_requests=time_e2e_requests,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

    # TODO: we may not need to decode
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Decodes the new token for a sequence."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        (new_tokens, new_output_text, prefix_offset, read_offset) = detokenize_incrementally(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.get_tokenizer_for_seq(seq),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            all_input_ids=seq.get_token_ids(),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prev_tokens=seq.tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prefix_offset=seq.prefix_offset,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            read_offset=seq.read_offset,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            skip_special_tokens=prms.skip_special_tokens,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            spaces_between_special_tokens=prms.spaces_between_special_tokens,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if seq.tokens is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq.tokens = new_tokens
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq.tokens.extend(new_tokens)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq.prefix_offset = prefix_offset
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq.read_offset = read_offset
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        seq.output_text += new_output_text

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Stop the finished sequences."""
        # for stop_str in sampling_params.stop:
        #     if seq.output_text.endswith(stop_str):
        #         self._finalize_sequence(seq, sampling_params, stop_str)
        #         seq.status = SequenceStatus.FINISHED_STOPPED
        #         return
        # if seq.get_last_token_id() in sampling_params.stop_token_ids:
        #     stop_str = self.get_tokenizer_for_seq(seq).convert_ids_to_tokens(seq.get_last_token_id())
        #     self._finalize_sequence(seq, sampling_params, stop_str)
        #     seq.status = SequenceStatus.FINISHED_STOPPED
        #     return

        # Check if the sequence has reached max_model_len.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if seq.get_len() > self.scheduler_config.max_model_len:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # Check if the sequence has reached max_tokens.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if seq.get_output_len() == sampling_params.max_tokens:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # Check if the sequence has generated the EOS token.
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if ((not sampling_params.ignore_eos) and
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                seq.get_last_token_id() == self.get_tokenizer_for_seq(seq).eos_token_id):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq.status = SequenceStatus.FINISHED_STOPPED
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _finalize_sequence(self, seq: Sequence, sampling_params: SamplingParams, stop_string: str) -> None:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if not sampling_params.include_stop_str_in_output and stop_string:
            # Truncate the output text so that the stop string is
            # not included in the output.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            seq.output_text = seq.output_text[:-len(stop_string)]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.add_lora(lora_request)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def remove_lora(self, lora_id: int) -> bool:
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert lora_id > 0, "lora_id must be greater than 0."
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.remove_lora(lora_id)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def list_loras(self) -> List[int]:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.worker.list_loras()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor]) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.sync_model_weights(actor_weights=actor_weights)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def offload_model_weights(self) -> None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.worker.offload_model_weights()


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
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        A tuple of (`distributed_init_method`, `placement_group`). The
        `distributed_init_method` is the address for initializing the
        distributed backend. `placement_group` includes the specification
        of the resources for each distributed worker.
    """

    # Initialize cluster locally.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    port = get_open_port()
    # We need to setup the distributed init method to make sure
    # the distributed megatron code (e.g., get world size) works correctly.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    distributed_init_method = f"tcp://localhost:{port}"
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return distributed_init_method, None


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_open_port():
    # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        s.bind(("", 0))
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return s.getsockname()[1]
