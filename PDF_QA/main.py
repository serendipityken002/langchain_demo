import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import chainlit as cl

llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")

@cl.on_chat_start
async def start():
    # cl.info("基于Chainlit实现PDF问答机器人")
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="请上传你要提问的PDF文件",
            accept=["application/pdf"], # 只接受PDF文件
            max_size_mb=20, # 限制文件大小为20MB
        ).send()

    _file = files[0]
    msg = cl.Message(content="正在加载PDF文件{_file.name}，请稍等...")
    await msg.send()

    filepath = f"../tmp/{_file.name}"
    if not os.path.exists("../tmp"):
        os.makedirs("../tmp")
    with open(filepath, "wb") as f:
        f.write(_file.content)
