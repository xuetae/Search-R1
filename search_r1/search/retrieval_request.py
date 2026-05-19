"""
检索请求客户端 (Retrieval Request Client)
===================================
功能：向检索服务器发送HTTP请求的示例客户端代码

使用方法：
- 设置检索服务器URL
- 准备查询列表和参数
- 发送POST请求
- 获取并处理返回的检索结果
"""

import requests  # HTTP请求库

# 本地FastAPI检索服务器的URL
url = "http://127.0.0.1:8000/retrieve"

# 请求数据载荷
payload = {
    "queries": ["What is the capital of France?", "Explain neural networks."] * 200,  # 查询列表
    "topk": 5,  # 返回前K个文档
    "return_scores": True  # 是否返回相似度分数
}

# 发送POST请求
response = requests.post(url, json=payload)

# 如果请求失败则抛出异常
response.raise_for_status()

# 获取JSON响应
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)
