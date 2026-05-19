# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import requests
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from fastapi import FastAPI
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pydantic import BaseModel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Optional, Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from concurrent.futures import ThreadPoolExecutor
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import uvicorn

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser = argparse.ArgumentParser(description="Launch online search server.")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--search_url', type=str, required=True, 
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    help="URL for search engine (e.g. https://serpapi.com/search)")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--topk', type=int, default=3, 
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    help="Number of results to return per query")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--serp_api_key', type=str, default=None, 
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    help="SerpAPI key for online search")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--serp_engine', type=str, default="google", 
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    help="SerpAPI engine for online search")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
args = parser.parse_args()

# --- Config ---
# 中文注释：下一行定义类，用于组织相关状态与行为。
class OnlineSearchConfig:
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        search_url: str = "https://serpapi.com/search",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        topk: int = 3,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        serp_api_key: Optional[str] = None,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        serp_engine: Optional[str] = None,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.search_url = search_url
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.topk = topk
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.serp_api_key = serp_api_key
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.serp_engine = serp_engine


# --- Online Search Wrapper ---
# 中文注释：下一行定义类，用于组织相关状态与行为。
class OnlineSearchEngine:
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: OnlineSearchConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _search_query(self, query: str):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        params = {
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "engine": self.config.serp_engine,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "q": query,
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            "api_key": self.config.serp_api_key,
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        }
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        response = requests.get(self.config.search_url, params=params)
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return response.json()

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def batch_search(self, queries: List[str]):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with ThreadPoolExecutor() as executor:
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for result in executor.map(self._search_query, queries):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                results.append(self._process_result(result))
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _process_result(self, search_result: Dict):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        answer_box = search_result.get('answer_box', {})
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if answer_box:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            title = answer_box.get('title', 'No title.')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            snippet = answer_box.get('snippet', 'No snippet available.')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.append({
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        organic_results = search_result.get('organic_results', [])
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for _, result in enumerate(organic_results[:self.config.topk]):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            title = result.get('title', 'No title.')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            snippet = result.get('snippet', 'No snippet available.')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.append({
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })

        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        related_results = search_result.get('related_questions', [])
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for _, result in enumerate(related_results[:self.config.topk]):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            title = result.get('question', 'No title.')  # question is the title here
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            snippet = result.get('snippet', 'No snippet available.')
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.append({
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            })

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return results


# --- FastAPI Setup ---
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
app = FastAPI(title="Online Search Proxy Server")

# 中文注释：下一行定义类，用于组织相关状态与行为。
class SearchRequest(BaseModel):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    queries: List[str]

# Instantiate global config + engine
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
config = OnlineSearchConfig(
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    search_url=args.search_url,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    topk=args.topk,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    serp_api_key=args.serp_api_key,
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    serp_engine=args.serp_engine,
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
engine = OnlineSearchEngine(config)

# --- Routes ---
# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@app.post("/retrieve")
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def search_endpoint(request: SearchRequest):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results = engine.batch_search(request.queries)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {"result": results}

## return {"result": List[List[{'document': {"id": xx, "content": "title" + \n + "content"}, 'score': xx}]]}

# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    uvicorn.run(app, host="0.0.0.0", port=8000)
