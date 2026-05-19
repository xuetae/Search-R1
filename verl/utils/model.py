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
模型工具 (Model Utils)
======================
功能：从Hugging Face加载和创建模型的通用工具

主要功能：
1. 创建Actor模型（LLM，用于生成）
2. 创建Critic模型（LLM + 价值头）
3. 配置模型覆盖
4. 模型大小计算

支持的模型：
- 所有AutoModelForCausalLM支持的模型
- 如：LLaMA, GPT, Mistral等
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 操作系统接口
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings  # 警告管理
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict, Type  # 类型注解

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy数组库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch import nn  # 神经网络模块
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, MistralForSequenceClassification  # Hugging Face
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.models.registry import ModelRegistry  # 模型注册表


# 中文注释：下一行定义类，用于组织相关状态与行为。
class LambdaLayer(nn.Module):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Lambda层：包装任意函数为PyTorch模块
    
    功能：将Python函数转换为可调用的神经网络层
    
    用途：在模型中集成自定义操作
    
    例子：
        # 创建一个squeeze层
        squeeze_layer = LambdaLayer(lambda x: torch.squeeze(x, dim=-1))
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, fn):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        super().__init__()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.fn = fn  # 保存函数

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def forward(self, *args, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """前向传播：调用包装的函数"""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.fn(*args, **kwargs)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def squeeze(x):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    压缩最后一个维度
    
    参数：
        x: 张量
        
    返回：
        压缩后的张量
    """
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.squeeze(x, dim=-1)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_model_config(module_config, override_config_kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    更新模型配置
    
    参数：
        module_config: 模型配置对象
        override_config_kwargs: 要覆盖的配置字典
        
    说明：
    - 直接修改module_config对象的属性
    - 用于在模型创建前调整配置参数
    """
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in override_config_kwargs.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        setattr(module_config, key, val)  # 设置配置属性


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> Dict:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    获取Actor模型配置
    
    参数：
        model_name: 模型名称或路径
        override_config_kwargs: 配置覆盖字典
        trust_remote_code: 是否信任远程代码
        
    返回：
        更新后的模型配置对象
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if override_config_kwargs is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config_kwargs = {}
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    
    # 从Hugging Face加载配置
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    
    # 应用配置覆盖
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    update_model_config(module_config, override_config_kwargs)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return module_config


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    创建Actor模型（生成模型）
    
    功能：创建用于生成文本的LLM模型
    
    参数：
        model_name: 模型名称或路径（如'meta-llama/Llama-2-7b'）
        override_config_kwargs: 配置覆盖（如num_hidden_layers=12）
        automodel_kwargs: AutoModelForCausalLM的参数（如device_map='auto'）
        
    返回：
        初始化的模型对象
        
    例子：
        actor = create_huggingface_actor(
            'meta-llama/Llama-2-7b',
            override_config_kwargs={'num_layers': 12},
            automodel_kwargs={'torch_dtype': torch.float16}
        )
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if override_config_kwargs is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        override_config_kwargs = {}
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if automodel_kwargs is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        automodel_kwargs = {}
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    
    # 获取配置
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    module_config = get_huggingface_actor_config(model_name,
                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                 override_config_kwargs,
                                                 # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                 trust_remote_code=automodel_kwargs.get('trust_remote_code', False))
    
    # 从配置创建模型
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return module


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    创建Critic模型（价值评估模型）
    
    功能：创建用于价值评估的模型
    
    架构：
    - 基础：Actor模型（LLM）
    - 修改：用价值头替换语言模型头
        - 输入：隐藏状态 [batch_size, seq_len, hidden_size]
        - 输出：价值分数 [batch_size]
    
    参数：
        model_name: 模型名称
        override_config_kwargs: 配置覆盖
        automodel_kwargs: 模型参数
        
    返回：
        初始化的Critic模型
    """
    # 基于Actor创建模型
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    critic_module: nn.Module = create_huggingface_actor(model_name,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        override_config_kwargs=override_config_kwargs,
                                                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                                        automodel_kwargs=automodel_kwargs)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if automodel_kwargs is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        automodel_kwargs = {}
    
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch_dtype = automodel_kwargs.get('torch_dtype', torch.float32)
    
    # 替换语言模型头为价值头
    # 价值头：线性层将隐藏状态映射到单个价值分数，然后squeeze
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    critic_module.lm_head = nn.Sequential(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype),  # 线性投影到1维
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        LambdaLayer(fn=squeeze)  # 压缩维度
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return critic_module


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_model_size(model: nn.Module, scale='auto'):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    n_params = sum(p.numel() for p in model.parameters())

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if scale == 'auto':
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if n_params > 1e9:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scale = 'B'
        # 中文注释：下一行继续判断其他条件分支。
        elif n_params > 1e6:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scale = 'M'
        # 中文注释：下一行继续判断其他条件分支。
        elif n_params > 1e3:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scale = 'K'
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            scale = ''

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if scale == 'B':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        n_params = n_params / 1e9
    # 中文注释：下一行继续判断其他条件分支。
    elif scale == 'M':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        n_params = n_params / 1e6
    # 中文注释：下一行继续判断其他条件分支。
    elif scale == 'K':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        n_params = n_params / 1e3
    # 中文注释：下一行继续判断其他条件分支。
    elif scale == '':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pass
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
        raise NotImplemented(f'Unknown scale {scale}')

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return n_params, scale


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def print_model_size(model: nn.Module, name: str = None):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    n_params, scale = get_model_size(model, scale='auto')
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if name is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        name = model.__class__.__name__
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f'{name} contains {n_params:.2f}{scale} parameters')


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def create_random_mask(input_ids: torch.Tensor,
                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                       max_ratio_of_valid_token: float,
                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                       max_ratio_of_left_padding: float,
                       # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                       min_ratio_of_valid_token: float = 0):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """Create a random mask given input_ids. Support left padding and right padding.
    Process:
    - Sample valid token length
    - Sample left_padding length
    - Generate padding

    Args:
        input_ids:
            shape (batch_size, seq_len)

    Returns:

    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size, sequence_length = input_ids.shape
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    masks = torch.ones_like(input_ids, dtype=torch.int64)
    # TODO: we can make this faster
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for i in range(batch_size):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for index in range(num_left_padding):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            masks[i, index] = 0

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for index in range(num_left_padding + num_valid, sequence_length):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            masks[i, index] = 0
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return masks


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def compute_position_id_with_mask(mask):
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def normalize_pp_vpp_params(params, num_hidden_layers, layer_name='layers'):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    Normalize the pp vpp params into a complete named parameters. 
    This is useful when gather parameters from pp ranks and passed to a model without pp

    params: List[List[Dict[str, param]]]
        params contains a list of pp, with a list of vpp named_parameters in each vpp chunk.
    output: Dict[str, param]

    """

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_layers):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Transform the model name in each model_chunk in each pp stage into the name in inference engine
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if vpp_size > 1:
            # print(f'try to bind vpp params to inference engine...')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layers_per_pp = num_layers // pp_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layers_per_vpp = layers_per_pp // vpp_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pp_offset = layers_per_vpp * pp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            vpp_offset = (layers_per_vpp * pp_size) * vpp_rank
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_offset = pp_offset + vpp_offset
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layers_per_pp = num_layers // pp_size
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_offset = layers_per_pp * pp_rank

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if layer_name in name:  # belong to an intermediate layer
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            split_name = name.split('.')
            # find the num next to split_name
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i, name in enumerate(split_name):
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if name == layer_name:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    break
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            layer_num_idx = i + 1
            # check the name
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(split_name) >= layer_num_idx + 1, f'split_name = {split_name}'
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert split_name[layer_num_idx].isdigit(), f'split_name = {split_name}'
            # increment layer_num_idx by layer_offset
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            name = '.'.join(split_name)  # weight name in inference_tp_model
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return name

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pp_size = len(params)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    normalized_name_to_param = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for pp_rank in range(len(params)):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        vpp_size = len(params[pp_rank])
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for vpp_rank in range(vpp_size):
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for name, param in params[pp_rank][vpp_rank].items():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                normalized_name = normalize_model_name(name, pp_rank, vpp_rank, pp_size, vpp_size, num_hidden_layers)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                normalized_name_to_param[normalized_name] = param

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return normalized_name_to_param


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_parallel_model_from_config(config, megatron_config, pre_process=None, post_process=None, value=False):
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from megatron.core import ModelParallelConfig
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(megatron_config, ModelParallelConfig)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_class = _get_parallel_model_architecture_from_config(config, value)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model = model_class(config, megatron_config, pre_process=pre_process, post_process=post_process)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return model


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def _get_parallel_model_architecture_from_config(config: PretrainedConfig, value=False) -> Type[nn.Module]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    architectures = getattr(config, "architectures", [])
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arch in architectures:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model_cls = ModelRegistry.load_model_cls(arch, value)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if model_cls is not None:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return model_cls
    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def load_megatron_model_weights(config,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                model_config,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                parallel_model,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                params_dtype,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                is_value_model=False,
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                local_cache_path='~/.cache/verl/rlhf'):
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    architectures = getattr(model_config, "architectures", [])
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    local_cache_path = os.path.expanduser(local_cache_path)

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if config.model.path.startswith("hdfs:"):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from verl.utils.fs import copy_local_path_from_hdfs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'start download from {config.model.path}')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_model_path = copy_local_path_from_hdfs(src=config.model.path, cache_dir=local_cache_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print('finish download')
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f"load from local dir {config.model.path}")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        local_model_path = config.model.path

    # TODO: to find a better way to load mistral7b-rm lm_head
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if 'mistral7b-rm' in config.model.path:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = MistralForSequenceClassification.from_pretrained(local_model_path)  # use score head instead of lm_head
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state_dict = model.state_dict()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state_dict['lm_head.weight'] = state_dict['score.weight']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state_dict['model.embed_tokens.weight'] = state_dict[
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'model.embed_tokens.weight'][:32000]  # workaround, 32001 -> 32000
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_value_model = True
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with warnings.catch_warnings():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            warnings.simplefilter("ignore")
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        state_dict = model.state_dict()

    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from verl.models.weight_loader_registry import get_weight_loader
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(f'before weight loader: architectures = {architectures}...')
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for arch in architectures:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(f'call weight loader arch = {arch}, model config = {model.config}')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_loader = get_weight_loader(arch)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        weight_loader(state_dict=state_dict,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      wrapped_models=parallel_model,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      config=model.config,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      params_dtype=params_dtype,
                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                      is_value_model=is_value_model)


# pad input_ids_rmpad, cu_seqlens and max_seqlen_in_batch to be divisible by tp
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """pad the tokens such that the total length is a multiple of size.
    This function is useful when applying sequence parallel and context parallel

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding
        cu_seqlens: (total_nnz + 1,)
        max_seqlen_in_batch: int

    Returns:

    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    F = nn.functional

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    total_nnz = unpad_tokens.shape[0]

    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if total_nnz % size == 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_size = 0
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_size = size - total_nnz % size

    # we assume adding a new data in the batch with seqlen pad_size
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pad_size > 0:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if unpad_tokens.ndim == 1:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        # 中文注释：下一行继续判断其他条件分支。
        elif unpad_tokens.ndim == 2:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise NotImplementedError(f'Padding dim {unpad_tokens.ndim()} is not supported')

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return unpad_tokens, cu_seqlens, max_seqlen_in_batch
