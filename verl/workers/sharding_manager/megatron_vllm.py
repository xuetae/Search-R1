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
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed as dist

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import parallel_state as mpu
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core import DistributedDataParallel as LocalDDP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from megatron.core.transformer.module import Float16Module
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.megatron_utils import get_model, unwrap_model
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.memory_buffer import (
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    build_memory_buffer,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    build_memory_reference_from_module,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    get_weight_buffer_meta_from_module,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
)


# 中文注释：下一行定义类，用于组织相关状态与行为。
class AllGatherPPModel:

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, model_provider) -> None:

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._pp_group = mpu.get_pipeline_model_parallel_group()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._pp_size = mpu.get_pipeline_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._model_chunk_size = self._vpp_size or 1

        # each one holds a list of model_chunks in this pp stage
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._pp_models = [None] * self.pp_size

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank_list = list(range(self.pp_size))
        # make current rank the last one to initialize
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank_list[self.pp_rank], rank_list[-1] = rank_list[-1], rank_list[self.pp_rank]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._this_rank_models = None

        # store the parameter of each pp stage
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.memory_buffers = [None] * self.pp_size
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for cur_pp_rank in rank_list:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'create pp model', f'torch allocated {torch.cuda.memory_allocated() / 1e9:.4f} GB, '
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'reserved {torch.cuda.memory_reserved() / 1e9:.4f} GB')
            # since the last initialized rank is the current pp rank, after init, the pp rank is still correct
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if cur_pp_rank != self.pp_rank:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                models = get_model(model_provider, wrap_with_ddp=False)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                models = nn.ModuleList(models)
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.pp_models[cur_pp_rank] = models
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # for regular model, we wrapped it with DDP
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                models = get_model(model_provider)
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert len(models) == self._model_chunk_size, f"{len(models)} != {self._model_chunk_size}"
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._this_rank_models = nn.ModuleList(models)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.pp_models[cur_pp_rank] = nn.ModuleList(unwrap_model(models, (torchDDP, LocalDDP)))

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._build_param_buffer(cur_pp_rank)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self._build_param_references(cur_pp_rank, maintain_weight=cur_pp_rank == self.pp_rank)

            # TODO: after binding to the memory buffer, we can load the checkpoint here
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if cur_pp_rank != self.pp_rank:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for model in self.pp_models[cur_pp_rank]:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    model.eval()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._offload_params_to_cpu(cur_pp_rank)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_param_buffer(self, pp_rank):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Build the parameter buffer in each pp rank"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = self.pp_models[pp_rank]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_buffer_meta = get_weight_buffer_meta_from_module(model)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.memory_buffers[pp_rank] = build_memory_buffer(weight_buffer_meta)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _build_param_references(self, pp_rank, maintain_weight=False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = self.pp_models[pp_rank]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        build_memory_reference_from_module(model, self.memory_buffers[pp_rank], maintain_weight=maintain_weight)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _load_params_to_cuda(self, pp_rank, to_empty=False):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert pp_rank != self.pp_rank, f"unexpected to load current pp rank [{pp_rank}] back to cuda"
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for buffer in self.memory_buffers[pp_rank].values():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not to_empty:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                buffer.data = buffer.data.to(torch.cuda.current_device(), non_blocking=True)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                buffer.data = torch.empty_like(buffer.data, device='cuda')
        # rebuild reference after loading to CUDA
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._build_param_references(pp_rank)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _offload_params_to_cpu(self, pp_rank, to_empty=False):
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert pp_rank != self.pp_rank, f"unexpected to offload current pp rank [{pp_rank}] to cpu"
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for buffer in self.memory_buffers[pp_rank].values():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not to_empty:
                # offload the whole memory buffer to CPU
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                buffer.data = buffer.data.to('cpu', non_blocking=True)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                buffer.data = torch.empty_like(buffer.data, device='cpu')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._build_param_references(pp_rank)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_params_to_cuda(self, to_empty=False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """load all model params to cuda"""
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for cur_pp_rank in range(self.pp_size):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if cur_pp_rank != self.pp_rank:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._load_params_to_cuda(cur_pp_rank, to_empty=to_empty)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def allgather_params(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """allgather params of all pp ranks. Return a list of handles"""
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for cur_pp_rank in range(self.pp_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            global_src = dist.get_global_rank(group=self.pp_group, group_rank=cur_pp_rank)

            # NOTE(sgm): the async op may cause memory leakage of the memory_buffer/pp_models
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for memory_buffer in self.memory_buffers[cur_pp_rank].values():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dist.broadcast(tensor=memory_buffer.data, src=global_src, group=self.pp_group, async_op=False)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self, *inputs, **kwargs):
        # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
        try:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            prev_output = None
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for cur_chunk_rank in range(self._model_chunk_size):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if self._vpp_size:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    mpu.set_virtual_pipeline_model_parallel_rank(cur_chunk_rank)

                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for cur_pp_rank in range(self.pp_size):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    mpu.set_pipeline_model_parallel_rank(cur_pp_rank)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(prev_output)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    ret = self.pp_models[cur_pp_rank][cur_chunk_rank](*inputs, **kwargs)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    self.pp_models[cur_pp_rank][cur_chunk_rank].set_input_tensor(None)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    prev_output = ret
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        finally:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if self._vpp_size:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                mpu.set_virtual_pipeline_model_parallel_rank(0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            mpu.set_pipeline_model_parallel_rank(self.pp_rank)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return ret

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self, *inputs, **kwargs):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.forward(*inputs, **kwargs)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def eval(self):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for model in self.pp_models[self.pp_rank]:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model.eval()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def train(self):
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for model in self.pp_models[self.pp_rank]:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            model.train()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def offload_params_to_cpu(self, to_empty=False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """offload params of models that are not of current pp rank to cpu"""
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for cur_pp_rank in range(self.pp_size):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if cur_pp_rank != self.pp_rank:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self._offload_params_to_cpu(cur_pp_rank, to_empty=to_empty)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get_all_params(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Get all the parameters of the models in all pp ranks

        Returns:
            params: List[List[Dict[str, Tensor]]]: a list of parameters in all pp, where each is a list of dict
                tensors of each model chunk

        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for pp_rank in range(self.pp_size):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            params.append([])
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for model_chunk_idx in range(len(self.pp_models[pp_rank])):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                params[pp_rank].append({})
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pp_model = self.pp_models[pp_rank][model_chunk_idx]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pp_model = unwrap_model(pp_model, ((torchDDP, LocalDDP, Float16Module)))  # not use Float16Module
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for name, param in pp_model.named_parameters():
                    # NOTE(gh) workaround: should not get lora params for inference
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if 'lora' in name:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        continue
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    params[pp_rank][model_chunk_idx][name] = param

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return params

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def update_this_rank_models(self, new_models):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._this_rank_models = new_models
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._pp_models[self.pp_rank] = unwrap_model(new_models, (torchDDP, LocalDDP))

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def this_rank_models(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._this_rank_models

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pp_size(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._pp_size

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pp_rank(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._pp_rank

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pp_group(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._pp_group

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @property
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pp_models(self):
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._pp_models


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .base import BaseShardingManager

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch.distributed
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.distributed import new_group

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import verl.utils.megatron.tensor_parallel as tp_utils
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.third_party.vllm import parallel_state as vllm_ps
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.third_party.vllm import LLM
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.model import normalize_pp_vpp_params
# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
_MICRO_DATA_PARALLEL_GROUP = None


# 中文注释：下一行定义类，用于组织相关状态与行为。
class MegatronVLLMShardingManager(BaseShardingManager):

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, module: AllGatherPPModel, inference_engine: LLM, model_config, layer_name_mapping):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module = module
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine = inference_engine
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.model_config = model_config
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.layer_name_mapping = layer_name_mapping

        # initialize micro_dp group for vllm inference
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        global _MICRO_DATA_PARALLEL_GROUP
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        world_size = torch.distributed.get_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rank = torch.distributed.get_rank()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        train_tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        infer_tensor_parallel_size = vllm_ps.get_tensor_model_parallel_world_size()

        # TODO(sgm): this may not be true for FSDP -> vLLM
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert infer_tensor_parallel_size <= train_tensor_parallel_size, \
            'Not implemented for infer_tp > train_tp'
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert train_tensor_parallel_size % infer_tensor_parallel_size == 0

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_dp_size = train_tensor_parallel_size // infer_tensor_parallel_size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_micro_dp_groups = world_size // micro_dp_size
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(num_micro_dp_groups):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            group = new_group(ranks=ranks)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if rank in ranks:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                _MICRO_DATA_PARALLEL_GROUP = group

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def default_tp_concat_fn(self, name, param, infer_params, model_config):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        name: name of the parameter
        param: training parameters
        infer_params (List[torch.Tensor]): a list of parameters all-gathered from micro_dp_group
        model_config: huggingface model_config
        TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
        definition so that it is model-agnostic. If the model doesn't implement this function, 
        we can throw an error to force user disable TP HybridEngine.
        """

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.layer_name_mapping.get("qkv_layer_name") in name:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            q_lst = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            k_lst = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            v_lst = []
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            num_q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for infer_param in infer_params:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                q, k, v = infer_param.split(split_size)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                q_lst.append(q)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                k_lst.append(k)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                v_lst.append(v)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            q = torch.cat(q_lst, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            k = torch.cat(k_lst, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            v = torch.cat(v_lst, dim=0)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            infer_params = torch.cat((q, k, v), dim=0)

        # 中文注释：下一行继续判断其他条件分支。
        elif self.layer_name_mapping.get("gate_proj_layer_name") in name:
            # if the tensor is gate and proj
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gate_lst = []
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            up_lst = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for infer_param in infer_params:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gate, up = infer_param.chunk(2)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                gate_lst.append(gate)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                up_lst.append(up)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gate = torch.cat(gate_lst, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            up = torch.cat(up_lst, dim=0)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            infer_params = torch.cat((gate, up), dim=0)

        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # concat tensor
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(param))

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return infer_params

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _post_process_params(self, params):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        For each param, if it is a tp-splited param, we all-gather from micro_dp group.
        """
        # here the params are in train tp format. we iterate params and all-gather
        # TODO(zhangchi.usc1992) We can consider copy non-tp weight to another infer buffer.
        # In this way, all the params in the original memory_buffers and can be offload.
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_dp_size = get_micro_data_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_dp_group = get_micro_data_parallel_group()

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if micro_dp_size <= 1:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        origin_params = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for name in params.keys():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            param = params[name]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if tp_utils.is_tensor_parallel_param(param):
                # allocate a new tensor with proper size
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                infer_params = [torch.empty_like(param) for _ in range(micro_dp_size)]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                torch.distributed.all_gather(infer_params, param, group=micro_dp_group)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                infer_params = self.default_tp_concat_fn(name, param, infer_params, self.model_config)
                # replace with original param
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                params[name] = infer_params
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            origin_params[name] = param

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return origin_params

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __enter__(self):
        # create a new cuda space for parameters not in this pp rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.load_params_to_cuda()
        # broadcast the parameters from pp rank to other ranks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.allgather_params()
        # obtain name to parameters in pp/vpp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params = self.module.get_all_params()

        # bind the params to inference engine
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.params = normalize_pp_vpp_params(params=params,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              num_hidden_layers=self.model_config.num_hidden_layers,
                                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                              layer_name='layers')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.origin_params = self._post_process_params(self.params)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine.sync_model_weights(self.params, load_format='megatron')

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __exit__(self, exc_type, exc_value, traceback):
        # offload parameters doesn't belong to this pp rank
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.offload_params_to_cpu()

        # FIXME(sgm): the best practice is to delete the cuda tensor
        # rebind the model weights, can be any cpu tensor
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if get_micro_data_parallel_world_size() > 1:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for name in self.params.keys():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                self.params[name] = self.origin_params[name]

        # self.inference_engine.sync_model_weights(params)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.inference_engine.offload_model_weights()

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.module.train()

        # add empty cache after each compute
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.cuda.empty_cache()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def preprocess_data(self, data: DataProto) -> DataProto:
        # prompts are identical for each training tp. We select for each inference tp
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_dp_size = get_micro_data_parallel_world_size()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_dp_rank = get_micro_data_parallel_rank()

        # broadcast from tp=0 to other tp ranks
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        broadcast_dict_tensor(data.batch,
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              src=mpu.get_tensor_model_parallel_src_rank(),
                              # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                              group=mpu.get_tensor_model_parallel_group())

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if micro_dp_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            local_prompts = data.chunk(chunks=micro_dp_size)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = local_prompts[micro_dp_rank]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def postprocess_data(self, data: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info = data.meta_info
        # all gather batch among micro-dp groups
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        micro_dp_size = get_micro_data_parallel_world_size()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if micro_dp_size > 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                size=get_micro_data_parallel_world_size(),
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                group=get_micro_data_parallel_group(),
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                dim=0)

        # all gather batch among pp group
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if meta_info.get('allgather_pp_output', True):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                size=mpu.get_pipeline_model_parallel_world_size(),
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                group=mpu.get_pipeline_model_parallel_group(),
                                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                dim=0)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return data


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Micro Data parallel group
"""


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_micro_data_parallel_group():
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert _MICRO_DATA_PARALLEL_GROUP is not None
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _MICRO_DATA_PARALLEL_GROUP


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_micro_data_parallel_world_size():
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.distributed.get_world_size(group=get_micro_data_parallel_group())


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_micro_data_parallel_rank():
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.distributed.get_rank(group=get_micro_data_parallel_group())
