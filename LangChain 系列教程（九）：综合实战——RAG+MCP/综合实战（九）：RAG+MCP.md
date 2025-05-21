# 综合实战（九）：多模态RAG问答系统

## 需求
|序号|核心需求|功能描述|
|-|-|-|
|1|本地知识库问答与信息检索|系统应支持加载用户提供的本地文档（如 PDF、DOCX、TXT 等），建立语义索引，并基于文档内容提供高效、准确的问答与信息检索服务。|
|2|实时外部信息的获取与整合|系统应具备在线检索能力，能够根据用户请求获取互联网上的相关内容，辅助回答问题或补充信息来源。|
|3|定向网页内容的抓取与处理|系统应能根据用户提供的网页链接抓取其主要内容，并按需完成摘要提取、内容分析或保存操作。|
|4|本地文件的自动化写入与管理|系统应支持在授权目录下执行文件写入、目录创建等操作，用于保存用户指定的内容或生成结果。|
|5|多轮对话中的上下文理解与管理|系统应具备对话记忆能力，能够在多轮交互中正确理解用户的指代与意图，保持响应的连贯性与上下文相关性。|
|6|自然语言指令的理解与任务协调执行|系统应能够解析复杂的自然语言指令，自动拆解任务并协调调用内部模块，完成多步骤复合操作。|


## 用到的MCP

|序号|MCP|说明|
|-|-|-|
|1|**Tavily**|提供搜索能力，调用 `search` 工具实现多轮网页检索接口，通过 MCP 协议封装为异步工具，支持 query 和 search_depth 等参数配置。用于查找论文或问题相关链接和摘要。|
|2|**Fetch**|提供 URL 内容抓取能力（支持网页或 PDF），调用 `fetch` 工具将外部资源解析为结构化文本内容。支持指定最大长度，结果由 Agent 读取处理。|
|3|**Filesystem**|提供文件系统访问能力，MCP 工具中包含 `write_file`、`list_directory` 等操作。允许 Agent 将生成的内容（如 Markdown 格式的论文摘要）写入本地白名单路径下的目录。|


## 代码
```python
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
```

```python
# ---------------------------------------------------------------------------
# 依赖、环境变量导入
# ---------------------------------------------------------------------------
import os, asyncio
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.tools import Tool

from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# 扫描本地知识库目录，批量加载 PDF / DOCX / TXT
# ---------------------------------------------------------------------------
BASE_DIR = "Docs"
documents = []
for fn in os.listdir(BASE_DIR):
    fp = os.path.join(BASE_DIR, fn)
    if fn.lower().endswith(".pdf"):
        documents.extend(PyPDFLoader(fp).load())
    elif fn.lower().endswith(".docx"):
        documents.extend(Docx2txtLoader(fp).load())
    elif fn.lower().endswith(".txt"):
        documents.extend(TextLoader(fp).load())

# ---------------------------------------------------------------------------
# 文本切分，构建 Qdrant 向量数据库
# ---------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks   = splitter.split_documents(documents)

DB_PATH = "./Docs-database"
COLL    = "my_documents"
emb     = OpenAIEmbeddings()

if not os.path.exists(DB_PATH):
    vectorstore = Qdrant.from_documents(chunks, emb, path=DB_PATH, collection_name=COLL)
else:
    client      = QdrantClient(path=DB_PATH)
    vectorstore = Qdrant(client=client, collection_name=COLL, embeddings=emb)

# ---------------------------------------------------------------------------
# 检索链：Multi-Query + ConversationalRetrieval
# ---------------------------------------------------------------------------
llm        = ChatOpenAI(model_name="deepseek-v3", temperature=0)
retriever  = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), llm)
memory     = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain   = ConversationalRetrievalChain.from_llm(llm, retriever, memory, verbose=True)

rag_tool = Tool(
    name="RAG_QA",
    func=lambda q: qa_chain({"question": q})["answer"],
    description="基于本地知识库的语义问答"
)

# ---------------------------------------------------------------------------
# MCP 工具配置：Tavily 搜索 / Fetch 抓取 / Filesystem 文件系统
# ---------------------------------------------------------------------------
MCP_CONFIGS = {
    "tavily": {
        "command": "python",
        "args": ["tavily_mcp.py"],
        "transport": "stdio"
    },
    "fetch": {
        "command": "uvx",
        "args": ["mcp-server-fetch"]
    },
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "/Users/orzjh/Desktop",
            "/Users/orzjh/Desktop/knowledge-base"
        ]
    },
}

# ---------------------------------------------------------------------------
# ask() —— 单轮调用，无记忆功能
# ---------------------------------------------------------------------------
async def ask(message: str) -> str:
    tools, cleanup = await convert_mcp_to_langchain_tools(MCP_CONFIGS)
    tools.append(rag_tool)
    try:
        agent = create_react_agent(llm, tools, checkpointer=InMemorySaver())
        resp  = await agent.ainvoke({"messages": message}, config={"thread_id": "session-001"})
        return resp["messages"][-1].content
    finally:
        await cleanup()

# ---------------------------------------------------------------------------
# main() —— 交互式命令行循环
# ---------------------------------------------------------------------------
async def main():
    tools, cleanup = await convert_mcp_to_langchain_tools(MCP_CONFIGS)
    tools.append(rag_tool)

    try:
        agent = create_react_agent(llm, tools, checkpointer=InMemorySaver())
        print("输入问题，exit/quit 退出。")
        while True:
            msg = input("You: ")
            if msg.lower() in {"exit", "quit"}:
                print("再见！")
                break
            resp = await agent.ainvoke({"messages": msg}, config={"thread_id": "session-001"})
            print("AI:", resp["messages"][-1].content)
    finally:
        await cleanup()

# ---------------------------------------------------------------------------
# CLI 启动入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
```

## 参考

