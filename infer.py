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
import transformers  # 模型加载库（AutoTokenizer, AutoModelForCausalLM）
import torch  # PyTorch张量操作库
import random  # 随机数生成库
from datasets import load_dataset  # 数据集加载库
import requests  # HTTP请求库（调用检索服务）

# ==================== 配置参数 ====================
# 测试问题 - 这是一个多跳问答的例子（需要搜索多次才能得到答案）
question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"

# 模型ID - 使用Qwen2.5微调后的Search-R1模型
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"

# 设备选择 - 优先使用GPU，回退到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 问题预处理 ====================
# 清除问题的空格
question = question.strip()
# 如果问题不以?结尾，添加?
if question[-1] != '?':
    question += '?'

# Qwen2.5系列模型的EOS令牌ID
curr_eos = [151645, 151643]  # 结束令牌ID列表
# 搜索结果的格式模板
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# ==================== 生成提示词 ====================
# 系统提示词 - 定义模型的行为规则
prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# ==================== 加载模型和分词器 ====================
# 初始化分词器（用于将文本转换为令牌）
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# 加载模型（bfloat16精度用于节省内存，device_map='auto'用于多GPU分布）
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,  # 使用混合精度（brain float 16）
    device_map="auto"  # 自动选择GPU分布
)

# ==================== 自定义停止条件 ====================
class StopOnSequence(transformers.StoppingCriteria):
    """
    自定义停止标准
    ===============
    当模型输出包含特定序列（如</search>或</answer>）时停止生成
    
    这允许模型在给出完整答案或提出搜索查询后立即停止，
    而不是生成直到max_new_tokens
    """
    def __init__(self, target_sequences, tokenizer):
        """
        初始化停止条件
        
        参数：
            target_sequences: 字符串列表，包含要查找的停止序列
            tokenizer: 分词器（用于将字符串转换为令牌ID）
        """
        # 将字符串序列编码为令牌ID列表
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False) 
            for target_sequence in target_sequences
        ]
        # 记录每个目标序列的长度（用于后续匹配）
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """
        检查是否应该停止生成
        
        参数：
            input_ids: 当前生成的令牌ID [batch_size, seq_len]
            scores: 模型的logits分数
            
        返回：
            True如果应该停止生成，False如果继续
        """
        # 将目标ID移到与input_ids相同的设备上（GPU/CPU）
        targets = [
            torch.as_tensor(target_id, device=input_ids.device) 
            for target_id in self.target_ids
        ]

        # 如果当前生成长度小于最短目标长度，肯定没有匹配，继续生成
        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # 检查input_ids的末尾是否与任何目标序列匹配
        for i, target in enumerate(targets):
            # 获取input_ids末尾与目标长度相同的部分
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                # 找到匹配，停止生成
                return True

        return False

def get_query(text):
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
    import re
    # 正则表达式：匹配<search>...</search>之间的任何内容（包括换行）
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    # 返回最后一个匹配（如果有多个搜索）
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
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
    payload = {
            "queries": [query],  # 查询列表
            "topk": 3,  # 返回前3个结果
            "return_scores": True  # 返回相关性分数
        }
    # 发送HTTP POST请求到本地检索服务
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        """
        将检索结果转换为格式化字符串
        
        参数：
            retrieval_result: 检索结果列表，每个元素是一个文档对象
            
        返回：
            格式化的字符串，包含文档编号、标题和内容
        """
        format_reference = ''  # 初始化输出字符串
        # 遍历每个检索到的文档
        for idx, doc_item in enumerate(retrieval_result):
            # 获取文档内容
            content = doc_item['document']['contents']
            # 第一行是标题（用\n分割）
            title = content.split("\n")[0]
            # 剩余部分是文本内容
            text = "\n".join(content.split("\n")[1:])
            # 格式化并添加到输出
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    # 获取第一个查询的结果并格式化
    return _passages2string(results[0])


# ==================== 初始化停止条件 ====================
# 定义停止序列 - 包含可能的变体（带空格、换行等）
target_sequences = [
    "</search>",  # 基本停止序列
    " </search>",  # 前面有空格
    "</search>\n",  # 后面有换行
    " </search>\n",  # 前后都有
    "</search>\n\n",  # 后面有两个换行
    " </search>\n\n"  # 前面空格+后面两个换行
]
# 创建停止条件对象列表
stopping_criteria = transformers.StoppingCriteriaList([
    StopOnSequence(target_sequences, tokenizer)
])

# ==================== 主推理循环 ====================
# 计数器，用于限制搜索轮数（防止无限循环）
cnt = 0

# 如果分词器有chat模板，应用它（将提示转换为chat格式）
if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        add_generation_prompt=True, 
        tokenize=False
    )

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(prompt)  # 打印初始提示

# 主循环 - 一直运行直到模型给出最终答案
while True:
    # 将提示编码为令牌ID
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    # 创建全1的注意力掩码（所有令牌都应该被注意）
    attention_mask = torch.ones_like(input_ids)
    
    # 调用模型进行生成
    outputs = model.generate(
        input_ids,  # 输入令牌ID
        attention_mask=attention_mask,  # 注意力掩码
        max_new_tokens=1024,  # 最多生成1024个新令牌
        stopping_criteria=stopping_criteria,  # 使用自定义停止条件
        pad_token_id=tokenizer.eos_token_id,  # 填充令牌ID
        do_sample=True,  # 使用采样而不是贪心解码
        temperature=0.7  # 温度参数（控制生成多样性）
    )

    # 检查生成是否以EOS令牌结尾
    if outputs[0][-1].item() in curr_eos:
        # 提取新生成的令牌（排除输入部分）
        generated_tokens = outputs[0][input_ids.shape[1]:]
        # 将令牌解码为文本
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # 打印最终输出
        print(output_text)
        # 如果是结束令牌，退出循环
        break

    # 提取新生成的令牌
    generated_tokens = outputs[0][input_ids.shape[1]:]
    # 解码为文本
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 从输出中提取搜索查询
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    if tmp_query:
        # 存在搜索查询，调用搜索服务
        search_results = search(tmp_query)
    else:
        # 没有搜索查询
        search_results = ''

    # 格式化搜索结果并添加到提示
    search_text = curr_search_template.format(
        output_text=output_text, 
        search_results=search_results
    )
    # 更新提示以包含新的搜索结果
    prompt += search_text
    # 增加计数
    cnt += 1
    # 打印当前轮次的搜索结果
    print(search_text)
