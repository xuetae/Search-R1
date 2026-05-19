# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Search-R1 LLM生成管理模块
============================
功能：负责LLM的推理与搜索交织的生成逻辑，包括：
1. 多轮交互生成（思考、搜索、回答）
2. 张量操作与批处理
3. 与环境交互执行预测
4. 多GPU并行处理

主要类：GenerationConfig, LLMGenerationManager
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch 张量计算库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re  # 正则表达式，用于解析搜索查询和答案
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from collections import defaultdict  # 默认字典，便于统计
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os  # 操作系统接口
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Dict, Any, Tuple  # 类型注解
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from dataclasses import dataclass  # 数据类装饰器
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from .tensor_helper import TensorHelper, TensorConfig  # 张量处理工具
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl import DataProto  # veRL框架的数据协议
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from verl.utils.tracking import Tracking  # 轨迹跟踪工具
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import shutil  # 文件操作库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import requests  # HTTP请求库

# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@dataclass
# 中文注释：下一行定义类，用于组织相关状态与行为。
class GenerationConfig:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    生成配置数据类
    ================
    包含生成过程中的所有超参数配置
    
    属性：
        max_turns: 最大交互轮数 (思考+搜索+回答的轮数)
        max_start_length: 提示词的最大长度
        max_prompt_length: 包括所有内容的最大提示长度
        max_response_length: 单次生成响应的最大长度
        max_obs_length: 观察信息（搜索结果）的最大长度
        num_gpus: 用于生成的GPU数量
        no_think_rl: 是否禁用思考模式（仅保留动作）
        search_url: 搜索服务的URL端点
        topk: 检索返回的前K个文档
    """
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_turns: int  # 最多执行多少轮的搜索/答案循环
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_start_length: int  # 初始提示的最大令牌数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_prompt_length: int  # 当前上下文的最大令牌数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_response_length: int  # 单次生成的最大令牌数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    max_obs_length: int  # 搜索结果观察的最大令牌数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    num_gpus: int  # 用于并行生成的GPU数量
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    no_think_rl: bool=False  # 是否不进行思考RL训练
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    search_url: str = None  # 搜索服务的HTTP地址
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    topk: int = 3  # 每次搜索返回的文档数

# 中文注释：下一行定义类，用于组织相关状态与行为。
class LLMGenerationManager:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    LLM生成管理器
    =============
    管理LLM的多轮推理与搜索交织的生成过程
    
    主要功能：
    1. 批量处理多个样本的生成
    2. 协调LLM生成、搜索、环境交互
    3. 处理张量的并行化和同步
    4. 管理多轮对话状态
    
    工作流程：
    for each turn:
        LLM生成 -> 解析<search>或<answer>标签 -> 执行行为 
        -> 获取观察 -> 更新上下文 -> 下一轮
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenizer,  # 分词器（将文本转换为令牌）
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actor_rollout_wg,  # Actor回滚工作组（执行LLM生成）
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        config: GenerationConfig,  # 生成配置对象
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        is_validation: bool = False,  # 是否处于验证模式
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化生成管理器
        
        参数：
            tokenizer: 用于编码/解码文本的分词器
            actor_rollout_wg: 执行LLM推理的计算单元
            config: 包含所有超参数的配置对象
            is_validation: 是否为验证阶段（可能有不同的行为）
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tokenizer = tokenizer  # 保存分词器
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.actor_rollout_wg = actor_rollout_wg  # 保存Actor工作组
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config  # 保存配置
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.is_validation = is_validation  # 保存验证标志

        # 初始化张量处理工具，配置令牌长度限制
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.tensor_fn = TensorHelper(TensorConfig(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pad_token_id=tokenizer.pad_token_id,  # 填充令牌ID
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_prompt_length=config.max_prompt_length,  # 最大提示长度
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_obs_length=config.max_obs_length,  # 最大观察长度
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            max_start_length=config.max_start_length  # 最大起始长度
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Tokenize a batch of responses."""
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self.tokenizer(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses, 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            add_special_tokens=False, 
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return_tensors='pt', 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            padding="longest"
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )['input_ids']

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Process responses to stop at search operation or answer operation."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses_str = self.tokenizer.batch_decode(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses, 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            skip_special_tokens=True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses_str = [resp.split('</search>')[0] + '</search>'
                 # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                 if '</search>' in resp 
                 # 中文注释：下一行处理前面条件都不满足时的默认分支。
                 else resp.split('</answer>')[0] + '</answer>'
                 # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                 if '</answer>' in resp 
                 # 中文注释：下一行处理前面条件都不满足时的默认分支。
                 else resp
                 # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                 for resp in responses_str]

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.no_think_rl:
            # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actions, _ = self.env.postprocess_predictions(responses_str)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print("RESPONSES:", responses_str)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        responses = self._batch_tokenize(responses_str)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return responses, responses_str

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Process next observations from environment."""
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        next_obs_ids = self.tokenizer(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            next_obs, 
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            padding='longest',
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return_tensors='pt',
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            add_special_tokens=False,  # Prevents adding special tokens
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )['input_ids']

        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return next_obs_ids

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            next_obs_ids: torch.Tensor) -> Dict:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollings.batch['input_ids'],
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            cur_responses,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            next_obs_ids
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ])
        
        # Create attention mask and position ids
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        effective_len = new_attention_mask.sum(dim=1).max()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_len = min(self.config.max_prompt_length, effective_len)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_rollings = DataProto.from_dict({
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'input_ids': new_input_ids[:, -max_len:],
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'position_ids': new_position_ids[:, -max_len:],
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            'attention_mask': new_attention_mask[:, -max_len:]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        })
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        new_rollings.meta_info.update(rollings.meta_info)
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return new_rollings

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _info_masked_concatenate_with_padding(self, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                prompt: torch.Tensor, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                prompt_with_mask: torch.Tensor, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                response: torch.Tensor, 
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                info: torch.Tensor = None,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pad_to_left: bool = True
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ) -> torch.Tensor:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_id = self.tokenizer.pad_token_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors = [prompt, response]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tensors_with_mask = [prompt_with_mask, response]
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if info is not None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensors.append(info)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tensors_with_mask.append(info_mask)
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        concatenated = torch.cat(tensors, dim=1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padded_tensor = concatenated.gather(1, sorted_indices)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return padded_tensor, padded_tensor_with_info

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _update_right_side(self, right_side: Dict, 
                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                          cur_responses: torch.Tensor,
                          # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                          next_obs_ids: torch.Tensor = None) -> Dict:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Update right side state."""
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if next_obs_ids != None:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    right_side['responses'],
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    right_side['responses_with_info_mask'],
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    cur_responses,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    next_obs_ids, 
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    pad_to_left=False
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    right_side['responses'],
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    right_side['responses_with_info_mask'],
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    cur_responses,
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    pad_to_left=False
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                )
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_len = min(self.config.max_prompt_length, effective_len)
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        num_gpus = self.config.num_gpus
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if num_gpus <= 1:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        batch_size = active_batch.batch['input_ids'].shape[0]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        remainder = batch_size % num_gpus
        
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in active_batch.batch.keys():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            active_batch.batch[key] = active_batch.batch[key].long()
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if remainder == 0:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padding_size = num_gpus - remainder
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padded_batch = {}
        
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padded_active_batch = DataProto.from_dict(padded_batch)
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for key in padded_active_batch.batch.keys():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            trimmed_meta = {}
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for k, v in padded_output.meta_info.items():
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if isinstance(v, torch.Tensor):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    trimmed_meta[k] = v[:-padding_size]
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    trimmed_meta[k] = v
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            padded_output.meta_info = trimmed_meta
            
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        padded_output.batch = trimmed_batch
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return padded_output

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Run main LLM generation loop."""
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        active_num_list = [active_mask.sum().item()]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        rollings = gen_batch

        # Main generation loop
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for step in range(self.config.max_turns):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not active_mask.sum():
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                break
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rollings.batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                keys=['input_ids', 'attention_mask', 'position_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollings_active = DataProto.from_dict({
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                k: v[active_mask] for k, v in rollings.batch.items()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info = gen_output.meta_info            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses_str, self.tokenizer.pad_token, active_mask
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            active_mask = active_mask * curr_active_mask
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            active_num_list.append(active_mask.sum().item())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            turns_stats[curr_active_mask] += 1
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollings = self._update_rolling_state(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rollings,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses_ids,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                next_obs_ids
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            original_right_side = self._update_right_side(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                original_right_side,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses_ids,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                next_obs_ids
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
            
        # final LLM rollout
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if active_mask.sum():
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                rollings.batch,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                keys=['input_ids', 'attention_mask', 'position_ids']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            rollings_active = DataProto.from_dict({
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                k: v[active_mask] for k, v in rollings.batch.items()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            meta_info = gen_output.meta_info            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            _, dones, valid_action, is_search = self.execute_predictions(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            active_mask = active_mask * curr_active_mask
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            active_num_list.append(active_mask.sum().item())
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            original_right_side = self._update_right_side(
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                original_right_side,
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                responses_ids,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            )
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info['turns_stats'] = turns_stats.tolist()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info['active_mask'] = active_mask.tolist()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _compose_final_output(self, left_side: Dict,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            right_side: Dict,
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """Compose final generation output."""
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output = right_side.copy()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output['input_ids'] = torch.cat([
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            left_side['input_ids'],
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            right_side['responses']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ], dim=1)
        
        # Create attention mask and position ids
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output['attention_mask'] = torch.cat([
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tensor_fn.create_attention_mask(final_output['responses'])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ], dim=1)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output['info_mask'] = torch.cat([
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ], dim=1)
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            final_output['attention_mask']
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        )
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output = DataProto.from_dict(final_output)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        final_output.meta_info.update(meta_info)
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return final_output

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        cur_actions, contents = self.postprocess_predictions(predictions)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if do_search:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            search_results = self.batch_search(search_queries)
            # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if not active:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                next_obs.append('')
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                dones.append(1)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                valid_action.append(0)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                is_search.append(0)
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if action == 'answer':
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    next_obs.append('')
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dones.append(1)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    valid_action.append(1)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    is_search.append(0)
                # 中文注释：下一行继续判断其他条件分支。
                elif action == 'search':
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dones.append(0)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    valid_action.append(1)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    is_search.append(1)
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    dones.append(0)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    valid_action.append(0)
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    is_search.append(0)
            
        # 中文注释：下一行进行运行时断言，确保关键前置条件成立。
        assert len(search_results) == 0
            
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return next_obs, dones, valid_action, is_search

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        actions = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        contents = []
                
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for prediction in predictions:
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if isinstance(prediction, str): # for llm output
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                pattern = r'<(search|answer)>(.*?)</\1>'
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                match = re.search(pattern, prediction, re.DOTALL)
                # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                if match:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    content = match.group(2).strip()  # Return only the content inside the tags
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    action = match.group(1)
                # 中文注释：下一行处理前面条件都不满足时的默认分支。
                else:
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    content = ''
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    action = None
            # 中文注释：下一行处理前面条件都不满足时的默认分支。
            else:
                # 中文注释：下一行主动抛出异常，提示调用方当前情况不可继续。
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            actions.append(action)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            contents.append(content)
            
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return actions, contents

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def batch_search(self, queries: List[str] = None) -> str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = self._batch_search(queries)['result']
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return [self._passages2string(result) for result in results]

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _batch_search(self, queries):
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        payload = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "queries": queries,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "topk": self.config.topk,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "return_scores": True
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return requests.post(self.config.search_url, json=payload).json()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _passages2string(self, retrieval_result):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        format_reference = ''
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx, doc_item in enumerate(retrieval_result):
            
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            content = doc_item['document']['contents']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            title = content.split("\n")[0]
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            text = "\n".join(content.split("\n")[1:])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return format_reference
