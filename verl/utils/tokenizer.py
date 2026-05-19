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
分词器工具 (Tokenizer Utils)
============================
功能：提供分词器加载和配置的通用工具

主要功能：
1. 从Hugging Face预加载分词器
2. 自动修正某些模型的特殊分词配置
3. 处理填充符号配置

支持的特殊处理：
- Gemma-2模型：修正EOS标记的歧义
- 自动设置填充符号ID
"""

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import warnings  # 警告管理

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
__all__ = ['hf_tokenizer']  # 导出的公共接口


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def set_pad_token_id(tokenizer):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    设置填充符号ID
    
    功能：如果pad_token_id为None，则使用eos_token_id作为替代
    
    参数：
        tokenizer: Hugging Face分词器对象
        
    说明：
    - 许多模型没有定义专用的填充符号
    - 使用EOS（End-of-Sequence）作为填充符号是常见做法
    - 添加警告信息以通知用户
    """
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tokenizer.pad_token_id is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tokenizer.pad_token is None:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer.pad_token = tokenizer.eos_token
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    创建Hugging Face预训练分词器
    
    参数：
        name_or_path: 分词器名称或本地路径
                      - 模型名称如'meta-llama/Llama-2-7b'
                      - 本地路径如'/path/to/tokenizer'
        correct_pad_token: 是否修正填充符号配置（默认True）
        correct_gemma2: 是否应用Gemma-2特殊处理（默认True）
        **kwargs: 其他传递给AutoTokenizer的参数
        
    返回：
        分词器对象
        
    说明：
    - Gemma-2修正：原模型中EOS标记存在歧义，可能影响RL性能
    - 自动修正缺少的填充符号配置
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    from transformers import AutoTokenizer
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if correct_gemma2 and isinstance(name_or_path, str) and 'gemma-2-2b-it' in name_or_path:
        # Gemma-2中的EOS标记存在歧义，这可能会降低RL性能
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        warnings.warn('Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.')
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kwargs['eos_token'] = '<end_of_turn>'
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        kwargs['eos_token_id'] = 107
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if correct_pad_token:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        set_pad_token_id(tokenizer)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return tokenizer