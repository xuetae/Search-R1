# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
"""
Search-R1推理脚本 - 端到端问答推理
===================================
功能：使用已训练的Search-R1模型进行推理
流程：
1. 加载模型和分词器
2. 为问题生成提示词
3. 执行多轮推理循环
4. 每轮可以选择搜索或给出答案
5. 搜索结果作为上下文反馈给模型
6. 最终输出答案

工作流程示意：
┌─────────────┐
│   问题      │
└──────┬──────┘
       │
       ├─> 模型思考 + 生成响应
       │    ├─ 搜索查询 -> 检索引擎获取文档 -> 更新上下文
       │    └─ 最终答案 -> 结束
       │
       └─> 输出答案

主要类/函数:
- StopOnSequence: 自定义停止条件（当检测到</search>或</answer>时停止）
- get_query: 从模型输出中提取搜索查询
- search: 调用检索服务获取文档
"""
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import transformers  # 模型加载库（AutoTokenizer, AutoModelForCausalLM）
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import torch  # PyTorch张量操作库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import random  # 随机数生成库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from datasets import load_dataset  # 数据集加载库
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import requests  # HTTP请求库（调用检索服务）

# ==================== 配置参数 ====================
# 测试问题 - 这是一个多跳问答的例子（需要搜索多次才能得到答案）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# 模型ID - 使用Qwen2.5微调后的Search-R1模型
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"

# 设备选择 - 优先使用GPU，回退到CPU
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 问题预处理 ====================
# 清除问题的空格
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
question = question.strip()
# 如果问题不以?结尾，添加?
# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if question[-1] != '?':
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    question += '?'

# Qwen2.5系列模型的EOS令牌ID
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
curr_eos = [151645, 151643]  # 结束令牌ID列表
# 搜索结果的格式模板
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# ==================== 生成提示词 ====================
# 系统提示词 - 定义模型的行为规则
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# ==================== 加载模型和分词器 ====================
# 初始化分词器（用于将文本转换为令牌）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# 加载模型（bfloat16精度用于节省内存，device_map='auto'用于多GPU分布）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
model = transformers.AutoModelForCausalLM.from_pretrained(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    model_id, 
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    torch_dtype=torch.bfloat16,  # 使用混合精度（brain float 16）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    device_map="auto"  # 自动选择GPU分布
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
)

# ==================== 自定义停止条件 ====================
# 中文注释：下一行定义类，用于组织相关状态与行为。
class StopOnSequence(transformers.StoppingCriteria):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    自定义停止标准
    ===============
    当模型输出包含特定序列（如</search>或</answer>）时停止生成
    
    这允许模型在给出完整答案或提出搜索查询后立即停止，
    而不是生成直到max_new_tokens
    """
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, target_sequences, tokenizer):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        初始化停止条件
        
        参数：
            target_sequences: 字符串列表，包含要查找的停止序列
            tokenizer: 分词器（用于将字符串转换为令牌ID）
        """
        # 将字符串序列编码为令牌ID列表
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.target_ids = [
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            tokenizer.encode(target_sequence, add_special_tokens=False) 
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for target_sequence in target_sequences
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]
        # 记录每个目标序列的长度（用于后续匹配）
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self._tokenizer = tokenizer

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __call__(self, input_ids, scores, **kwargs):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        检查是否应该停止生成
        
        参数：
            input_ids: 当前生成的令牌ID [batch_size, seq_len]
            scores: 模型的logits分数
            
        返回：
            True如果应该停止生成，False如果继续
        """
        # 将目标ID移到与input_ids相同的设备上（GPU/CPU）
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        targets = [
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            torch.as_tensor(target_id, device=input_ids.device) 
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for target_id in self.target_ids
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ]

        # 如果当前生成长度小于最短目标长度，肯定没有匹配，继续生成
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if input_ids.shape[1] < min(self.target_lengths):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return False

        # 检查input_ids的末尾是否与任何目标序列匹配
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for i, target in enumerate(targets):
            # 获取input_ids末尾与目标长度相同的部分
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                # 找到匹配，停止生成
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return True

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return False

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def get_query(text):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    从模型输出中提取搜索查询
    
    功能：使用正则表达式找到<search>和</search>之间的文本
    
    参数：
        text: 模型生成的完整文本
        
    返回：
        搜索查询字符串，如果没找到则返回None
        
    例子：
        输入: "Let me search. <search>Mike Barnett player</search>"
        输出: "Mike Barnett player"
    """
    # 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
    import re
    # 正则表达式：匹配<search>...</search>之间的任何内容（包括换行）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    matches = pattern.findall(text)
    # 返回最后一个匹配（如果有多个搜索）
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if matches:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return matches[-1]
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return None

# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def search(query: str):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    """
    调用检索服务搜索查询
    
    功能：
    1. 构建请求payload
    2. 发送HTTP请求到检索服务
    3. 解析返回的文档
    4. 格式化为字符串
    
    参数：
        query: 搜索查询字符串
        
    返回：
        格式化的文档字符串（包含标题和内容）
        
    服务格式：
        请求: POST http://127.0.0.1:8000/retrieve
        返回: {"result": [[{doc1}, {doc2}, ...], [...]]}
    """
    # 构建请求数据
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    payload = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "queries": [query],  # 查询列表
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "topk": 3,  # 返回前3个结果
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "return_scores": True  # 返回相关性分数
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
    # 发送HTTP POST请求到本地检索服务
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _passages2string(retrieval_result):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        """
        将检索结果转换为格式化字符串
        
        参数：
            retrieval_result: 检索结果列表，每个元素是一个文档对象
            
        返回：
            格式化的字符串，包含文档编号、标题和内容
        """
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        format_reference = ''  # 初始化输出字符串
        # 遍历每个检索到的文档
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for idx, doc_item in enumerate(retrieval_result):
            # 获取文档内容
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            content = doc_item['document']['contents']
            # 第一行是标题（用\n分割）
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            title = content.split("\n")[0]
            # 剩余部分是文本内容
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            text = "\n".join(content.split("\n")[1:])
            # 格式化并添加到输出
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return format_reference

    # 获取第一个查询的结果并格式化
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return _passages2string(results[0])


# ==================== 初始化停止条件 ====================
# 定义停止序列 - 包含可能的变体（带空格、换行等）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
target_sequences = [
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "</search>",  # 基本停止序列
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    " </search>",  # 前面有空格
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "</search>\n",  # 后面有换行
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    " </search>\n",  # 前后都有
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    "</search>\n\n",  # 后面有两个换行
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    " </search>\n\n"  # 前面空格+后面两个换行
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
]
# 创建停止条件对象列表
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
stopping_criteria = transformers.StoppingCriteriaList([
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    StopOnSequence(target_sequences, tokenizer)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
])

# ==================== 主推理循环 ====================
# 计数器，用于限制搜索轮数（防止无限循环）
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
cnt = 0

# 如果分词器有chat模板，应用它（将提示转换为chat格式）
# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if tokenizer.chat_template:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt = tokenizer.apply_chat_template(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        [{"role": "user", "content": prompt}], 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        add_generation_prompt=True, 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tokenize=False
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
print(prompt)  # 打印初始提示

# 主循环 - 一直运行直到模型给出最终答案
# 中文注释：下一行开始循环，直到条件不再满足。
while True:
    # 将提示编码为令牌ID
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    # 创建全1的注意力掩码（所有令牌都应该被注意）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    attention_mask = torch.ones_like(input_ids)
    
    # 调用模型进行生成
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    outputs = model.generate(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        input_ids,  # 输入令牌ID
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        attention_mask=attention_mask,  # 注意力掩码
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        max_new_tokens=1024,  # 最多生成1024个新令牌
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        stopping_criteria=stopping_criteria,  # 使用自定义停止条件
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        pad_token_id=tokenizer.eos_token_id,  # 填充令牌ID
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        do_sample=True,  # 使用采样而不是贪心解码
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        temperature=0.7  # 温度参数（控制生成多样性）
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )

    # 检查生成是否以EOS令牌结尾
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if outputs[0][-1].item() in curr_eos:
        # 提取新生成的令牌（排除输入部分）
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        generated_tokens = outputs[0][input_ids.shape[1]:]
        # 将令牌解码为文本
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # 打印最终输出
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        print(output_text)
        # 如果是结束令牌，退出循环
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        break

    # 提取新生成的令牌
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    generated_tokens = outputs[0][input_ids.shape[1]:]
    # 解码为文本
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 从输出中提取搜索查询
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
    if tmp_query:
        # 存在搜索查询，调用搜索服务
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        search_results = search(tmp_query)
    # 中文注释：下一行处理前面条件都不满足时的默认分支。
    else:
        # 没有搜索查询
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        search_results = ''

    # 格式化搜索结果并添加到提示
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    search_text = curr_search_template.format(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        output_text=output_text, 
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        search_results=search_results
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    )
    # 更新提示以包含新的搜索结果
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    prompt += search_text
    # 增加计数
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    cnt += 1
    # 打印当前轮次的搜索结果
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    print(search_text)
