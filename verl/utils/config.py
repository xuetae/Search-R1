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
配置工具 (Config Utils)
=======================
功能：配置对象管理和转换工具

主要功能：
- 从OmegaConf配置对象更新Python字典
- 支持配置和字典之间的同步
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import Dict  # 类型注解

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from omegaconf import DictConfig  # OmegaConf配置对象


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def update_dict_with_config(dictionary: Dict, config: DictConfig):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    使用配置对象更新字典
    
    功能：将DictConfig对象中的属性复制到Python字典中
    
    参数：
        dictionary: 待更新的Python字典
        config: OmegaConf DictConfig配置对象
        
    说明：
    - 只更新字典中已存在的键
    - 如果配置中有对应属性，则用配置值覆盖
    - 配置中不存在的键不会添加到字典中
    
    例子：
        dict = {'lr': 0.001, 'epochs': 10}
        cfg = DictConfig({'lr': 0.0001, 'batch_size': 32})
        update_dict_with_config(dict, cfg)
        # dict现在为 {'lr': 0.0001, 'epochs': 10}
    """
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for key in dictionary:
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hasattr(config, key):  # 检查配置中是否有该属性
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            dictionary[key] = getattr(config, key)  # 使用配置值覆盖
