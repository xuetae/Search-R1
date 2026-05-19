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


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_weight_loader(arch: str):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.models.llama.megatron.checkpoint_utils.llama_loader import load_state_dict_to_megatron_llama
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {'LlamaForCausalLM': load_state_dict_to_megatron_llama}

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     f"Supported architectures: {_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}")
