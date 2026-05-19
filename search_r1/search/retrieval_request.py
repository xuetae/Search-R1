"""
检索请求客户端 (Retrieval Request Client)
===================================
功能：向检索服务器发送HTTP请求的示例客户端代码

使用方法：
- 设置检索服务器URL
- 准备查询列表和参数
- 发送POST请求
- 获取并处理返回的检索结果
\"\"\"\n\nimport requests  # HTTP请求库\n\n# 本地FastAPI检索服务器的URL\nurl = \"http://127.0.0.1:8000/retrieve\"\n\n# 请求数据载荷\npayload = {\n    \"queries\": [\"What is the capital of France?\", \"Explain neural networks.\"] * 200,  # 查询列表\n    \"topk\": 5,  # 返回前K个文档\n    \"return_scores\": True  # 是否返回相似度分数\n}\n\n# 发送POST请求\nresponse = requests.post(url, json=payload)\n\n# 如果请求失败则抛出异常\nresponse.raise_for_status()\n\n# 获取JSON响应\nretrieved_data = response.json()\n\nprint(\"Response from server:\")\nprint(retrieved_data)
