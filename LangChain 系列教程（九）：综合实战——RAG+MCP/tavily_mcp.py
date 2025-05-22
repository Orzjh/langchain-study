# 
# tavily_mcp.py
# 将 Tavily 搜索能力封装为一个独立 MCP 工具服务器，能够响应来自智能 Agent 的检索请求，并返回格式化后的搜索结果。
# 

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
mcp = FastMCP("Tavily")

@mcp.tool()
async def search(query: str, search_depth: str = "basic") -> str:
    try:
        results = tavily.search(query=query, search_depth=search_depth, max_results=5)
        formatted = []
        
        for r in results.get("results", []):
            formatted.append(
                f"标题：{r.get('title', '无标题')}\n"
                f"内容：{r.get('content', '无内容')}\n"
                f"链接：{r.get('url', '无链接')}\n"
            )
        
        return "\n---\n".join(formatted) if formatted else "未找到相关结果"
    except Exception as e:
        return f"搜索出错：{str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")



