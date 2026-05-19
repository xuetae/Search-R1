# === 中文逐行注释辅助：本文件已按复现学习用途补充中文注释，原始代码逻辑保持不变。===
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import os
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import re
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import requests
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import argparse
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import asyncio
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import random
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from typing import List, Optional, Dict
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from concurrent.futures import ThreadPoolExecutor

# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import chardet
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import aiohttp
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import bs4
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
import uvicorn
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from fastapi import FastAPI
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from pydantic import BaseModel
# 中文注释：下一行导入依赖，为后续代码提供外部模块或工具函数。
from googleapiclient.discovery import build


# --- CLI Args ---
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser = argparse.ArgumentParser(description="Launch online search server.")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--api_key', type=str, required=True, help="API key for Google search")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--cse_id', type=str, required=True, help="CSE ID for Google search")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--topk', type=int, default=3, help="Number of results to return per query")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
parser.add_argument('--snippet_only', action='store_true', help="If set, only return snippets; otherwise, return full context.")
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
args = parser.parse_args()


# --- Config ---
# 中文注释：下一行定义类，用于组织相关状态与行为。
class OnlineSearchConfig:
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, topk: int = 3, api_key: Optional[str] = None, cse_id: Optional[str] = None, snippet_only: bool = False):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.topk = topk
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.api_key = api_key
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.cse_id = cse_id
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.snippet_only = snippet_only


# --- Utilities ---
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def parse_snippet(snippet: str) -> List[str]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    segments = snippet.split("...")
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return [s.strip() for s in segments if len(s.strip().split()) > 5]


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def sanitize_search_query(query: str) -> str:
    # Remove or replace special characters that might cause issues.
    # This is a basic example; you might need to add more characters or patterns.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sanitized_query = re.sub(r'[^\w\s]', ' ', query)  # Replace non-alphanumeric and non-whitespace with spaces.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sanitized_query = re.sub(r'[\t\r\f\v\n]', ' ', sanitized_query) # replace tab, return, formfeed, vertical tab with spaces.
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    sanitized_query = re.sub(r'\s+', ' ', sanitized_query).strip() #remove duplicate spaces, and trailing/leading spaces.

    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return sanitized_query


# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def filter_links(search_results: List[Dict]) -> List[str]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    links = []
    # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
    for result in search_results:
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for item in result.get("items", []):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if "mime" in item:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            ext = os.path.splitext(item["link"])[1]
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if ext in ["", ".html", ".htm", ".shtml"]:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                links.append(item["link"])
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return links


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore) -> str:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    user_agents = [
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P)...",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "Mozilla/5.0 AppleWebKit/537.36...",
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)",
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    ]
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    headers = {"User-Agent": random.choice(user_agents)}

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    async with semaphore:
        # 中文注释：下一行开始异常保护区域，捕获可能失败的操作。
        try:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            async with session.get(url, headers=headers) as response:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                raw = await response.read()
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                detected = chardet.detect(raw)
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                encoding = detected["encoding"] or "utf-8"
                # 中文注释：下一行返回当前函数的计算结果或控制信号。
                return raw.decode(encoding, errors="ignore")
        # 中文注释：下一行处理异常分支，保证错误可控。
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return ""


# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
async def fetch_all(urls: List[str], limit: int = 8) -> List[str]:
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    semaphore = asyncio.Semaphore(limit)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    timeout = aiohttp.ClientTimeout(total=5)
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    connector = aiohttp.TCPConnector(limit_per_host=limit, force_close=True)

    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        tasks = [fetch(session, url, semaphore) for url in urls]
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return await asyncio.gather(*tasks)


# --- Search Engine ---
# 中文注释：下一行定义类，用于组织相关状态与行为。
class OnlineSearchEngine:
    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def __init__(self, config: OnlineSearchConfig):
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        self.config = config

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def collect_context(self, snippet: str, doc: str) -> str:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        snippets = parse_snippet(snippet)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        ctx_paras = []

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for s in snippets:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            pos = doc.replace("\n", " ").find(s)
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if pos == -1:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                continue
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            sta = pos
            # 中文注释：下一行开始循环，直到条件不再满足。
            while sta > 0 and doc[sta] != "\n":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                sta -= 1
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            end = pos + len(s)
            # 中文注释：下一行开始循环，直到条件不再满足。
            while end < len(doc) and doc[end] != "\n":
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                end += 1
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            para = doc[sta:end].strip()
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if para not in ctx_paras:
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                ctx_paras.append(para)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return "\n".join(ctx_paras)

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def fetch_web_content(self, search_results: List[Dict]) -> Dict[str, str]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        links = filter_links(search_results)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        contents = asyncio.run(fetch_all(links))
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        content_dict = {}
        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for html, link in zip(contents, links):
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            soup = bs4.BeautifulSoup(html, "html.parser")
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            text = "\n".join([p.get_text() for p in soup.find_all("p")])
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            content_dict[link] = text
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return content_dict

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def search(self, search_term: str, num_iter: int = 1) -> List[Dict]:
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        service = build('customsearch', 'v1', developerKey=self.config.api_key)
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results = []
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        sanitize_search_term = sanitize_search_query(search_term)
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if search_term.isspace():
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return results
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        res = service.cse().list(q=sanitize_search_term, cx=self.config.cse_id).execute()
        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
        results.append(res)

        # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
        for _ in range(num_iter - 1):
            # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
            if 'nextPage' not in res.get('queries', {}):
                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                break
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            start_idx = res['queries']['nextPage'][0]['startIndex']
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            res = service.cse().list(q=search_term, cx=self.config.cse_id, start=start_idx).execute()
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            results.append(res)

        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return results

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def batch_search(self, queries: List[str]) -> List[List[str]]:
        # 中文注释：下一行进入上下文管理器，自动管理资源生命周期。
        with ThreadPoolExecutor() as executor:
            # 中文注释：下一行返回当前函数的计算结果或控制信号。
            return list(executor.map(self._retrieve_context, queries))

    # 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
    def _retrieve_context(self, query: str) -> List[str]:
        
        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
        if self.config.snippet_only:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            search_results = self.search(query)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            contexts = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for result in search_results:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for item in result.get("items", []):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    title = item.get("title", "")
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    context = ' '.join(parse_snippet(item.get("snippet", "")))
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if title != "" or context != "":
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        title = "No title." if not title else title
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        context = "No snippet available." if not context else context
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        contexts.append({
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            'document': {"contents": f'\"{title}\"\n{context}'},
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        })
        # 中文注释：下一行处理前面条件都不满足时的默认分支。
        else:
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            content_dict = self.fetch_web_content(search_results)
            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
            contexts = []
            # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
            for result in search_results:
                # 中文注释：下一行开始循环，逐项处理集合或批次中的元素。
                for item in result.get("items", []):
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    link = item["link"]
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    title = item.get("title", "")
                    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                    snippet = item.get("snippet", "")
                    # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                    if link in content_dict:
                        # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                        context = self.collect_context(snippet, content_dict[link])
                        # 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
                        if title != "" or context != "":
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            title = "No title." if not title else title
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            context = "No snippet available." if not context else context
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            contexts.append({
                                # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                                'document': {"contents": f'\"{title}\"\n{context}'},
                            # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
                            })
        
        # 中文注释：下一行返回当前函数的计算结果或控制信号。
        return contexts[:self.config.topk]


# --- FastAPI App ---
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
app = FastAPI(title="Online Search Proxy Server")

# 中文注释：下一行定义类，用于组织相关状态与行为。
class SearchRequest(BaseModel):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    queries: List[str]

# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
config = OnlineSearchConfig(api_key=args.api_key, cse_id=args.cse_id, topk=args.topk, snippet_only=args.snippet_only)
# 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
engine = OnlineSearchEngine(config)

# 中文注释：下一行是装饰器，用于给后续函数或类附加框架行为。
@app.post("/retrieve")
# 中文注释：下一行定义函数，封装当前模块中的一段可复用逻辑。
def search_endpoint(request: SearchRequest):
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    results = engine.batch_search(request.queries)
    # 中文注释：下一行返回当前函数的计算结果或控制信号。
    return {"result": results}


# 中文注释：下一行进入条件分支，根据运行状态选择不同处理路径。
if __name__ == "__main__":
    # 中文注释：下一行保持原始实现逻辑，是当前流程中的一个具体执行步骤。
    uvicorn.run(app, host="0.0.0.0", port=8000)
