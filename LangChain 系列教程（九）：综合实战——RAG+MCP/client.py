# 
# client.py
# 

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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

base_dir = 'Docs'
documents = []

for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

if not os.path.exists("./Docs-database"):
    vectorstore = Qdrant.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        path="./Docs-database",
        collection_name="my_documents"
    )
else:
    client = QdrantClient(path="./Docs-database")
    vectorstore = Qdrant(
        client=client,
        collection_name="my_documents",
        embeddings=OpenAIEmbeddings()
    )

llm         = ChatOpenAI(model_name="deepseek-v3", temperature=0)
retriever   = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
chat_memory = ConversationBufferMemory(memory="chat_history", return_messages=True)
qa_chain    = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=chat_memory, verbose=True)
rag_tool = Tool(name="RAG_QA", func=lambda q: qa_chain({"question": q})["answer"], description="doc QA")

mcp_configs = {
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

async def ask(msg):
    tools, cleanup = await convert_mcp_to_langchain_tools(mcp_configs)
    tools.append(rag_tool)
    try:
        agent = create_react_agent(llm, tools, checkpointer=InMemorySaver())
        res   = await agent.ainvoke({"messages": msg}, config={"thread_id": "session-001"})
        return res["messages"][-1].content
    finally:
        await cleanup()

async def main():
    tools, cleanup = await convert_mcp_to_langchain_tools(mcp_configs)
    tools.append(rag_tool)

    try:
        agent = create_react_agent(llm, tools, checkpointer=InMemorySaver())
        while True:
            msg = input("You: ")
            if msg.lower() in ("exit", "quit"):
                print("再见！")
                break

            res = await agent.ainvoke({"messages": msg}, config={"thread_id": "session-001"})
            print("AI response:", res["messages"][-1].content)
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())


# 抓取stable-diffusion这篇论文的完整内容pdf（摘要、介绍、主要方法等）并转化为markdown格式，保存到test文件夹下。回答用中文
# 抓取3d-gaussian-splatting这篇论文的完整内容pdf（摘要、介绍、主要方法等）并转化为markdown格式，保存到test文件夹下。回答用中文
# 抓取anydoor这篇论文的完整内容pdf（摘要、介绍、主要方法等）并转化为markdown格式，保存到test文件夹下。回答用中文
# 抓取https://leetcode.cn/problems/next-permutation/description/，保存到test文件夹下，然后给出解法。回答用中文
# 我刚才说了什么？