"""
张量处理助手模块
=================
功能：提供张量操作的便利函数，处理填充、长度切割、注意力掩码等
主要处理与LLM生成相关的张量操作，如上下文连接、长度管理等

主要类：TensorConfig, TensorHelper
"""
import torch  # PyTorch张量库
from typing import Dict, Tuple, List  # 类型注解
from dataclasses import dataclass  # 数据类装饰器

@dataclass
class TensorConfig:
    """
    张量配置数据类
    ================
    包含张量处理所需的配置参数
    
    属性：
        pad_token_id: 填充令牌的ID（用于代替无意义的位置）
        max_prompt_length: 提示的最大长度（超过此长度会被截断）
        max_obs_length: 观察信息的最大长度
        max_start_length: 起始提示的最大长度
    """
    pad_token_id: int  # 填充令牌ID，通常是0或分词器的特殊令牌
    max_prompt_length: int  # 提示的长度限制
    max_obs_length: int  # 观察结果（搜索结果）的长度限制
    max_start_length: int  # 初始提示的长度限制

class TensorHelper:
    """
    张量操作助手类
    ================
    提供与RL训练相关的张量操作，包括：
    1. 张量长度管理（截断、填充）
    2. 掩码生成（注意力掩码、位置ID）
    3. 张量连接和重新组织
    4. 填充结构转换
    """
    def __init__(self, config: TensorConfig):
        """
        初始化张量助手
        
        参数：
            config: TensorConfig对象，包含所有配置参数
        """
        self.config = config  # 保存配置

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor], 
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """
        根据注意力掩码将张量切割到有效长度
        
        功能：获取张量中真实数据的长度（排除填充），然后只保留那部分
        
        参数：
            tensor_dict: 包含各种张量的字典，必须包含'attention_mask'
            keys: 需要切割的张量的键名列表
            cut_left: 是否从左边截断（True则保留右边，False保留左边）
            
        返回：
            修改后的张量字典，其中指定的键已被切割
            
        示例：
            input_ids shape: (batch_size, seq_len)
            attention_mask: 表示哪些位置是真实数据（1）哪些是填充（0）
            输出：只保留有效部分
        """
        # 计算每个样本的有效长度（非填充令牌数）
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()  # 取批次中最长的有效长度
        result = tensor_dict.copy()  # 复制原字典
        
        # 对每个指定的键进行切割
        for key in keys:
            if cut_left:
                # 保留右边（最新的）的有效长度部分 [seq_len - effective_len:]
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                # 保留左边的有效长度部分 [:effective_len]
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        转换填充结构并返回排序后的张量和索引
        
        功能：将填充的位置移到左边或右边，用于高效的并行处理
        
        参数：
            tensor: 输入张量（可能有左右混合的填充）
            pad_to_left: 如果为True，将填充移到左边；如果为False，移到右边
            
        返回：
            (重新组织后的张量, 用于还原原始顺序的索引)
            
        例子：
            原始: [1, 2, PAD, 3, PAD, 4, 5]
            pad_to_left=True后: [PAD, PAD, 1, 2, 3, 4, 5]
            pad_to_left=False后: [1, 2, 3, 4, 5, PAD, PAD]
        """
        # 创建掩码：非填充位置为True（pad_to_left）或False（反向）
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        # 根据掩码排序，稳定排序以保持相对顺序
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        # 使用索引重新排列张量，返回重排后的张量和用于还原的索引
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        从input_ids创建注意力掩码
        
        功能：标记哪些位置应该被注意（真实令牌=1），哪些应该被忽略（填充令牌=0）
        
        参数：
            input_ids: 形状为(batch_size, seq_len)的令牌ID张量
            
        返回：
            注意力掩码张量，形状相同，1表示真实令牌，0表示填充
            
        例子：
            input_ids: [101, 2054, 0, 0]
            返回: [1, 1, 0, 0]
        """
        # 如果令牌ID不等于pad_token_id，掩码值为1；否则为0
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        从注意力掩码创建位置ID
        
        功能：为每个真实令牌分配连续的位置ID（0, 1, 2, ...），填充位置设为0
        
        参数：
            attention_mask: 注意力掩码，1表示真实位置
            
        返回：
            位置ID张量，形状与输入相同
            
        例子：
            attention_mask: [1, 1, 1, 0, 0]
            返回: [0, 1, 2, 0, 0]
        """
        # 累积求和得到位置，然后减1（因为cumsum从1开始）
        # 最后乘以attention_mask将填充位置的位置ID设为0
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor], 
                               pad_to_left: bool = True) -> torch.Tensor:
        """
        连接多个张量并处理填充
        
        功能：将多个张量沿序列维度连接，然后重新组织填充位置
        
        参数：
            tensors: 要连接的张量列表，都应该是(batch_size, seq_len)形状
            pad_to_left: 填充后是否将填充位置放在左边
            
        返回：
            连接后并重新组织的张量
            
        例子：
            张量1: [1, 2, PAD]
            张量2: [3, 4, 5]
            连接后: [1, 2, PAD, 3, 4, 5]
            重组后: [1, 2, 3, 4, 5, PAD] (if pad_to_left=False)
        """
        # 沿序列维度（dim=1）连接所有张量
        concatenated = torch.cat(tensors, dim=1)
        # 转换填充结构以优化内存访问
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor, 
                          responses_str: List[str], 
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        为非活跃样本填充响应
        
        功能：对于已完成（非活跃）的样本，用填充令牌替换其响应
        
        参数：
            responses: 只包含活跃样本响应的张量
            responses_str: 只包含活跃样本响应字符串的列表
            active_mask: 布尔掩码，表示哪些样本是活跃的
            
        返回：
            (填充后的响应张量, 填充后的响应字符串列表)
            批次大小恢复到原始大小
            
        例子：
            原始批次大小: 4
            活跃样本: 2个 (active_mask: [True, False, True, False])
            responses只有2个样本
            返回: 批次大小为4的响应（非活跃位置填充为空字符串）
        """
        # 验证活跃样本数与responses数量一致
        assert active_mask.sum() == responses.shape[0]
        # 获取原始批次大小和响应序列长度
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        # 创建全填充张量（形状为原始批次大小）
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        # 将活跃样本的响应放入对应位置
        padded_responses[active_mask] = responses
        
        # 为响应字符串做同样处理（使用空字符串作为填充）
        padded_responses_str = [""] * batch_size  # 初始化全空字符串列表
        
        # 遍历活跃样本并填充
        s = 0  # 活跃样本的索引
        for i, is_active in enumerate(active_mask):
            if is_active:  # 如果这个位置是活跃的
                padded_responses_str[i] = responses_str[s]  # 放入响应
                s += 1  # 移到下一个活跃样本
                
        return padded_responses, padded_responses_str  # 返回填充后的结果