from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import anyio  # 替换 asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_tools import convert_mcp_to_langchain_tools

load_dotenv()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model_name="gpt-4o")

mcp_configs = {
    "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"]
    },
    "tavily": {
        "command": "python",
        "args": ["tavily_mcp.py"],
        "transport": "stdio",
    },
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/orzjh/Desktop",
            "/Users/orzjh/Desktop/knowledge-base"
        ]
    }
}

async def run_agent(messages: str):
    async with anyio.create_task_group() as root_tg:
        # 在根任务组内初始化工具
        tools, cleanup = await convert_mcp_to_langchain_tools(mcp_configs)
        print(type(tools), tools)
        
        # 运行 Agent
        agent = create_react_agent(model, tools)
        agent_response = await agent.ainvoke({"messages": messages})
        
        # 清理资源
        await cleanup()
        return agent_response

if __name__ == "__main__":
    user_message = "抓取stable-diffusion这篇论文的内容并转化为markdown格式，保存到knowledge_base_sd文件夹下"
    result = anyio.run(run_agent, user_message)
    print("AI Response:\n", result["messages"][-1].content)