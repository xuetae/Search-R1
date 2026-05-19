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
veRL数据传输协议
================
功能：定义函数和模块之间的标准数据交换协议

核心概念：
- DataProto: 主要的数据容器，包含张量数据和元信息
- DataProtoItem: 单个数据点的表示
- TensorDict: PyTorch的张量字典结构，支持高效的批处理操作

设计特点：
1. 支持混合数据类型（张量和非张量数据）
2. 自动批处理管理
3. 支持灵活的数据索引和切片
4. 支持元信息存储（用于追踪数据源、权重等）

典型使用场景：
- 在Actor（生成）、Reward Model、Critic等组件间传输数据
- 管理RL轨迹数据（prompts、responses、rewards等）
- 支持分布式训练中的数据同步
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import pickle  # Python对象序列化
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import numpy as np  # NumPy数组库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import copy  # 深度复制
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass, field  # 数据类装饰器
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Callable, Dict, List, Union  # 类型注解

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch张量库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import tensordict  # PyTorch TensorDict库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from tensordict import TensorDict  # TensorDict类
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from torch.utils.data import DataLoader, Dataset  # 数据加载工具

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.py_functional import union_two_dict  # 字典合并工具

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['DataProto', 'union_tensor_dict']  # 导出接口

# 禁用TensorDict的遗留模式
# 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
try:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensordict.set_lazy_legacy(False).set()
# 中文注释：下一行处理异常分支，保证错误可控。
except:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pass


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def pad_dataproto_to_divisor(data: 'DataProto', size_divisor: int):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    填充DataProto使其大小可被指定整数整除
    
    功能：对于多GPU训练，批次大小需要能被GPU数整除
    
    参数：
        data: 待填充的DataProto
        size_divisor: 大小整除数（通常是GPU数量）
        
    返回：
        (填充后的DataProto, 填充的样本数)
        
    例子：
        原始批次大小: 10
        size_divisor: 4
        填充后: 12（需要添加2个样本）
    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert isinstance(data, DataProto), 'data must be a DataProto'
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(data) % size_divisor != 0:
        # 计算需要填充的样本数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_size = size_divisor - len(data) % size_divisor
        # 通过复制前pad_size个样本来填充
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_padded = DataProto.concat([data, data[:pad_size]])
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 无需填充
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_size = 0
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data_padded = data
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data_padded, pad_size


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def unpad_dataproto(data: 'DataProto', pad_size):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    移除填充的样本
    
    功能：与pad_dataproto_to_divisor相反
    
    参数：
        data: 包含填充的DataProto
        pad_size: 填充的样本数（来自pad_dataproto_to_divisor的返回值）
        
    返回：
        去除填充后的DataProto
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if pad_size != 0:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        data = data[:-pad_size]  # 移除最后pad_size个样本
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return data


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    合并两个TensorDict
    
    功能：将tensor_dict2中的所有键值对添加到tensor_dict1
    
    参数：
        tensor_dict1: 目标TensorDict（会被修改）
        tensor_dict2: 源TensorDict
        
    返回：
        合并后的TensorDict（与tensor_dict1相同）
        
    约束：
    - 两个字典必须有相同的批次大小
    - 如果键重复，值必须相同
    """
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, \
        f'Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}'
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key in tensor_dict2.keys():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if key not in tensor_dict1.keys():
            # 新键直接添加
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensor_dict1[key] = tensor_dict2[key]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 重复键需要验证值相同
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert tensor_dict1[key].equal(tensor_dict2[key]), \
                f'{key} in tensor_dict1 and tensor_dict2 are not the same object'

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return tensor_dict1


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    合并两个NumPy数组字典
    
    类似于union_tensor_dict但用于NumPy数组
    """
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in tensor_dict2.items():
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if key in tensor_dict1:
            # 验证类型
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert isinstance(tensor_dict2[key], np.ndarray)
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert isinstance(tensor_dict1[key], np.ndarray)
            # 验证值相同
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert np.all(tensor_dict2[key] == tensor_dict1[key]), \
                f'{key} in tensor_dict1 and tensor_dict2 are not the same object'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_dict1[key] = val

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return tensor_dict1


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    转换数据结构：从字典列表到列表字典
    
    功能：便于批处理
    
    例子：
        输入:  [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        输出:  {'a': [1, 3], 'b': [2, 4]}
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if len(list_of_dict) == 0:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {}
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    keys = list_of_dict[0].keys()  # 获取所有键
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output = {key: [] for key in keys}  # 初始化输出字典
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for data in list_of_dict:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, item in data.items():
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert key in output
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output[key].append(item)  # 添加到对应键的列表
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return output


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def fold_batch_dim(data: 'DataProto', new_batch_size):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    折叠批次维度
    
    功能：将[bsz, xxx]重形为[new_bsz, bsz // new_bsz, xxx]
    
    用途：用于某些特殊的处理流程（如多步RL）
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size = data.batch.batch_size[0]
    # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
    assert batch_size % new_batch_size == 0

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor: TensorDict = data.batch
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor = data.non_tensor_batch

    # 重形张量
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor = tensor.view(new_batch_size, -1)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor.auto_batch_size_(batch_dims=1)

    # 重形非张量数据
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in non_tensor.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return DataProto(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def unfold_batch_dim(data: 'DataProto', batch_dims=2):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    展开批次维度
    
    功能：与fold_batch_dim相反
    
    将[new_bsz, bsz // new_bsz, xxx]重形为[bsz, xxx]
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor: TensorDict = data.batch
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor = data.non_tensor_batch
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor.auto_batch_size_(batch_dims=batch_dims)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tensor = tensor.view(-1)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch_size = tensor.batch_size[0]

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor_new = {}
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in non_tensor.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return DataProto(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def collate_fn(x: list['DataProtoItem']):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    数据加载的整理函数
    
    功能：用于DataLoader，将多个DataProtoItem合并为单个DataProto
    
    参数：
        x: DataProtoItem列表
        
    返回：
        合并后的DataProto
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch = []
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor_batch = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for data in x:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch.append(data.batch)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor_batch.append(data.non_tensor_batch)
    # 堆叠张量
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch = torch.stack(batch).contiguous()
    # 转换非张量数据
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key, val in non_tensor_batch.items():
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor_batch[key] = np.array(val, dtype=object)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class DataProtoItem:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    单个数据点表示
    
    包含：
    - batch: 张量数据（TensorDict）
    - non_tensor_batch: 非张量数据（如字符串、对象等）
    - meta_info: 元信息（如数据源、ID等）
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch: TensorDict = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor_batch: Dict = field(default_factory=dict)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    meta_info: Dict = field(default_factory=dict)


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class DataProto:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    veRL数据协议 - 主要数据容器
    
    功能：标准化函数间的数据交换
    
    属性：
        batch: TensorDict，包含所有张量数据
               支持高效的批处理操作
        non_tensor_batch: 字典，包含非张量数据
               如答案、元数据等不适合张量的数据
        meta_info: 元信息字典
               用于追踪数据源、权重分数等
    
    设计理念：
    - TensorDict允许像操作单个Tensor一样操作Tensor字典
    - 支持自动批处理（batch_size、device管理）
    - 支持灵活的索引和切片操作
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    batch: TensorDict = None
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    non_tensor_batch: Dict = field(default_factory=dict)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    meta_info: Dict = field(default_factory=dict)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __post_init__(self):
        # 执行初始化后的一致性检查
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.check_consistency()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __len__(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """返回数据的样本数（批次大小）"""
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.batch is not None:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.batch.batch_size[0]
        # 中文注释：下一行继续判断其他条件分支。
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            # 如果没有张量，从非张量数据获取大小
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            random_key = list(self.non_tensor_batch.keys())[0]
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.non_tensor_batch[random_key].shape[0]
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return 0

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __getitem__(self, item):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """索引操作，返回DataProtoItem"""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_data = self.batch[item]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __getstate__(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """序列化支持"""
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import io
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer = io.BytesIO()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if tensordict.__version__ >= '0.5.0' and self.batch is not None:
            # 确保张量连续以提高I/O效率
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.batch = self.batch.contiguous()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.batch = self.batch.consolidate()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        torch.save(self.batch, buffer)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        buffer_bytes = buffer.getvalue()
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __setstate__(self, data):
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        import io
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch = torch.load(batch_deserialized,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           weights_only=False,
                           # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                           map_location='cpu' if not torch.cuda.is_available() else None)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch = batch
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.non_tensor_batch = non_tensor_batch
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.meta_info = meta_info

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def save_to_disk(self, filepath):
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with open(filepath, 'wb') as f:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pickle.dump(self, f)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def load_from_disk(filepath) -> 'DataProto':
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with open(filepath, 'rb') as f:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            data = pickle.load(f)
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return data

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def print_size(self, prefix=""):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        size_of_tensordict = 0
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, tensor in self.batch.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            size_of_tensordict += tensor.element_size() * tensor.numel()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        size_of_numpy_array = 0
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, numpy_array in self.non_tensor_batch.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            size_of_numpy_array += numpy_array.nbytes

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        size_of_numpy_array /= 1024**3
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        size_of_tensordict /= 1024**3

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        message = f'Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB'

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if prefix:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            message = f'{prefix}, ' + message
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(message)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def check_consistency(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.batch is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(self.batch.batch_size) == 1, 'only support num_batch_dims=1'

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.non_tensor_batch is not None:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for key, val in self.non_tensor_batch.items():
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert isinstance(val, np.ndarray)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(self.batch.batch_size) == 1, 'only support num_batch_dims=1 when non_tensor_batch is not empty.'

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_size = self.batch.batch_size[0]
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for key, val in self.non_tensor_batch.items():
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert isinstance(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    val, np.ndarray
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ) and val.dtype == object, 'data in the non_tensor_batch must be a numpy.array with dtype=object'
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert val.shape[
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    0] == batch_size, f'key {key} length {len(val)} is not equal to batch size {batch_size}'

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors = {}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensors = {}

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, val in data.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(val, torch.Tensor):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                tensors[key] = val
            # 中文注释：下一行继续判断其他条件分支。
            elif isinstance(val, np.ndarray):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                non_tensors[key] = val
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f'Unsupported type in data {type(val)}')

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @classmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(tensors) > 0, 'tensors must not be empty'
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert num_batch_dims > 0, 'num_batch_dims must be greater than zero'
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if non_tensors is not None:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert num_batch_dims == 1, 'only support num_batch_dims=1 when non_tensors is not None.'

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if meta_info is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info = {}
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if non_tensors is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensors = {}

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(non_tensors, dict)

        # get and check batch size
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = None
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pivot_key = None
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, tensor in tensors.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if batch_size is None:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                batch_size = tensor.shape[:num_batch_dims]
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pivot_key = key
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                current_batch = tensor.shape[:num_batch_dims]
                # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
                assert batch_size == current_batch, \
                    f'Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}'

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, val in non_tensors.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensors[key] = np.array(val, dtype=object)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def to(self, device) -> 'DataProto':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.batch is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.batch = self.batch.to(device)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> 'DataProto':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        # TODO (zhangchi.usc1992) whether to copy
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if batch_keys is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_keys = tuple(batch_keys)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sub_batch = self.batch.select(*batch_keys)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sub_batch = self.batch

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if non_tensor_batch_keys is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_batch = self.non_tensor_batch

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if deepcopy:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if meta_info_keys is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sub_meta_info = self.meta_info

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if deepcopy:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sub_meta_info = copy.deepcopy(sub_meta_info)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> 'DataProto':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert batch_keys is not None
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if meta_info_keys is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info_keys = []
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if non_tensor_batch_keys is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_batch_keys = []

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors = {}
        # tensor batch
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in batch_keys:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert key in self.batch.keys()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensors[key] = self.batch.pop(key)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensors = {}
        # non tensor batch
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in non_tensor_batch_keys:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert key in self.non_tensor_batch.keys()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensors[key] = self.non_tensor_batch.pop(key)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in meta_info_keys:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert key in self.meta_info.keys()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info[key] = self.meta_info.pop(key)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def rename(self, old_keys=None, new_keys=None) -> 'DataProto':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Note that this function only rename the key in the batch
        """

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def validate_input(keys):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if keys is not None:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if isinstance(keys, str):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    keys = [keys]
                # 中文注释：下一行继续判断其他条件分支。
                elif isinstance(keys, list):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    pass
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                    raise TypeError(f'keys must be a list or a string, but got {type(keys)}')
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return keys

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        old_keys = validate_input(old_keys)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_keys = validate_input(new_keys)

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if len(new_keys) != len(old_keys):
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                f'new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}')

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def union(self, other: 'DataProto') -> 'DataProto':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if
        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch = union_tensor_dict(self.batch, other.batch)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if dataloader_kwargs is None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dataloader_kwargs = {}

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if seed is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            generator = torch.Generator()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            generator.manual_seed(seed)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            generator = None

        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert isinstance(dataloader_kwargs, Dict)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        train_dataloader = DataLoader(dataset=self,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      batch_size=mini_batch_size,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      collate_fn=collate_fn,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      generator=generator,
                                      # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                      **dataloader_kwargs)

        # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
        def get_data():
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for _ in range(epochs):
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for d in train_dataloader:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    d.meta_info = self.meta_info
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    yield d

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return iter(get_data())

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def chunk(self, chunks: int) -> List['DataProto']:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.

        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self) % chunks == 0, f'only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}.'

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.batch is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_lst = [None for _ in range(chunks)]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor_batch_lst = [{} for _ in range(chunks)]
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, val in self.non_tensor_batch.items():
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert isinstance(val, np.ndarray)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_lst = np.array_split(val, chunks)
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(non_tensor_lst) == chunks
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for i in range(chunks):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(chunks):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output.append(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info))

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def concat(data: List['DataProto']) -> 'DataProto':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Concat a list of DataProto. The batch is concatenated among dim=0.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DataProto]): list of DataProto

        Returns:
            DataProto: concatenated DataProto
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for batch in data:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch_lst.append(batch.batch)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if batch_lst[0] is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_batch = torch.cat(batch_lst, dim=0)
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            new_batch = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, val in non_tensor_batch.items():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def reorder(self, indices):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Note that this operation is in-place
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        indices_np = indices.detach().numpy()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.batch = self.batch[indices]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def repeat(self, repeat_times=2, interleave=True):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.batch is not None:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if interleave:
                # Interleave the data
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                repeated_tensors = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # Stack the data
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                repeated_tensors = {
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                    for key, tensor in self.batch.items()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                }

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            repeated_batch = TensorDict(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                source=repeated_tensors,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            repeated_batch = None

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        repeated_non_tensor_batch = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key, val in self.non_tensor_batch.items():
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if interleave:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return DataProto(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            batch=repeated_batch,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            non_tensor_batch=repeated_non_tensor_batch,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info=self.meta_info,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )


# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import ray


# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class DataProtoFuture:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible. 
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    collect_fn: Callable
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    futures: List[ray.ObjectRef]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    dispatch_fn: Callable = None

    # 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
    @staticmethod
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def concat(data: List[ray.ObjectRef]) -> 'DataProtoFuture':
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def chunk(self, chunks: int) -> List['DataProtoFuture']:
        # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
        from functools import partial

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        arg_future_lst = []
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
            def dispatch_fn(x, i, chunks):
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return x.chunk(chunks=chunks)[i]

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            arg_future = DataProtoFuture(collect_fn=self.collect_fn,
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks),
                                         # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                         futures=self.futures)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            arg_future_lst.append(arg_future)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return arg_future_lst

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def get(self):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = ray.get(self.futures)  # dp_size.
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for o in output:
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert isinstance(o, DataProto)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output = self.collect_fn(output)  # select dp, concat
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.dispatch_fn is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return output
