import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
import logging
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 读取文件内容
file_path = "./novel/content_1746695216.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1 对长文本进行分词切割
text_spider = TokenTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
    encoding_name="gpt2"
)
docs = text_spider.create_documents([content])

# 2 将文档存储向量数据库
db = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.tech/v1"),
    persist_directory="./novel/chroma_db"
)

retriver = db.as_retriever(search_kwargs={"k": 2})

# query = "萧炎的师父是谁？"
# docs = retriver.invoke(query)
# for doc in docs:
#     print(doc.page_content)
#     print("-" * 50)

# retriver_from_llm = MultiQueryRetriever.from_llm(
#     retriever=retriver,
#     llm=ChatOpenAI(base_url="https://api.chatanywhere.tech/v1"),
# )

# # 设置日志级别
# logging.basicConfig()
# logging.getLogger(
#     'langchain.retrievers.multi_query'
# ).setLevel(logging.INFO)

# docs = retriver_from_llm.invoke("萧炎的师父是谁？")
# for doc in docs:
#     print(doc.page_content)
#     print("-" * 50)

# 3 检索向量数据库
compressor = LLMChainExtractor.from_llm(llm=ChatOpenAI(base_url="https://api.chatanywhere.tech/v1"))
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriver
)

docs = compression_retriever.invoke("药老让萧炎买哪些东西？")
for doc in docs:
    print(doc.page_content)
    print("-" * 50)