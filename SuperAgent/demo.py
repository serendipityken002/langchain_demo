import os
import re
import json
from dotenv import load_dotenv
from WebSpider_bilibili import main as bilibili_main
from WebSpider_biqugen import main as biqugen_main
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor, ZeroShotAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory

# 初始化环境
load_dotenv(dotenv_path="./api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# 全局状态管理
class AppState:
    def __init__(self):
        self.last_file_path = None
        self.last_db_path = None
        self.last_summary = None
        self.processing_history = []

    def add_history(self, action, details):
        self.processing_history.append({"action": action, "details": details})
    
    def get_history(self):
        return self.processing_history
    
app_state = AppState()

# 1. 爬虫工具函数
def spider_bilibili(url):
    """爬取B站视频内容"""
    app_state.add_history("爬取", f"开始爬取B站视频: {url}")
    res = bilibili_main(url)
    app_state.add_history("爬取", res)
    return res

def spider_biqugen(url):
    """爬取笔趣阁小说内容"""
    app_state.add_history("爬取", f"开始爬取笔趣阁小说: {url}")
    contents = biqugen_main(url, first=15)
    
    # 生成唯一文件名并保存内容
    import time
    file_path = f"./novel/content_{int(time.time())}.txt"
    os.makedirs("./novel", exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(contents)
    
    app_state.last_file_path = file_path
    app_state.add_history("爬取", f"笔趣阁小说爬取完成，保存到: {file_path}")
    
    return f"内容已保存到文件: {file_path}，总字符数: {len(contents)}"

# 2. 总结工具函数
def summarize_file_content(file_path=None):
    """总结文件内容"""
    # 如果没有提供文件路径，则使用上次保存的路径
    if not file_path and app_state.last_file_path:
        file_path = app_state.last_file_path
    if not file_path:
        return "错误：没有指定文件路径，也没有找到上次爬取的文件"
    
    app_state.add_history("总结", f"开始总结文件内容: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"错误：无法读取文件 {file_path}，请检查文件路径和权限。错误信息: {str(e)}"

    # 进行长文本总结
    summary = summarize_long_text(content)
    app_state.last_summary = summary
    app_state.add_history("总结", f"长文本总结完成") # 详细总结内容分开存储

    return summary

# 3. RAG 查询工具函数
def setup_rag_database(file_path=None):
    """设置RAG数据库"""
    if not file_path and app_state.last_file_path:
        file_path = app_state.last_file_path
    if not file_path:
        return "错误：没有指定文件路径，也没有找到上次爬取的文件"
    
    app_state.add_history("RAG", f"开始设置RAG数据库: {file_path}")
    
    # 1 读取文件内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"错误：无法读取文件 {file_path}，请检查文件路径和权限。错误信息: {str(e)}"
    
    # 2 进行文本分割
    text_splitter = TokenTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        encoding_name="gpt2"
    )
    docs = text_splitter.create_documents([content])
    
    # 3 将文档存储向量数据库
    db_path = "./novel/chroma_db"
    db = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.tech/v1"),
        persist_directory=db_path
    )

    app_state.last_db_path = db_path
    app_state.add_history("RAG", f"向量数据库建立完成，路径: {db_path}")

    return f"向量数据库已创建: {db_path}"

def query_rag_database(query, db_path=None):
    """在向量数据库中查询信息"""
    if not db_path and app_state.last_db_path:
        db_path = app_state.last_db_path
    
    if not db_path:
        return "错误：没有指定数据库路径，也没有找到上次创建的数据库"
    
    app_state.add_history("查询", f"RAG查询: '{query}'")
    
    # 加载向量数据库
    db = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.tech/v1")
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # 设置压缩器
    compressor = LLMChainExtractor.from_llm(llm=ChatOpenAI(base_url="https://api.chatanywhere.tech/v1"))
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    # 执行查询
    docs = compression_retriever.invoke(query)
    
    # 格式化结果
    results = []
    for doc in docs:
        results.append(doc.page_content)
    
    result_text = "\n\n".join(results)
    app_state.add_history("查询", "RAG查询完成")
    
    return result_text

# 实现长文本总结功能
def init_summary_chain(llm, verbose=False):
    # 创建内容分析和总结的提示模板
    summary_template = """
    你是一个专业的内容分析师。请分析以下内容，确定其类型（如视频内容、小说文本等），
    然后提供一个详尽的总结。如果内容过长，请提取关键信息。

    内容：
    {content}

    请提供以下分析：
    1. 关键要点：内容中最重要的3-5个要点是什么？
    2. 总结：用200字左右总结内容的精华，总结要求如下。
    - 必须严格依据原文内容，不能添加任何想象或修饰性语句（如"扣人心弦""人物形象丰满"等无实质内容的套话）。
    - 使用小说式语言，语言流畅自然，突出人物动机与冲突，展现情节逻辑。
    """

    summary_prompt = PromptTemplate(
        input_variables=["content"],
        template=summary_template
    )

    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        verbose=verbose
    )
    return summary_chain

def init_longtext_summary_chain(llm, verbose=False):
    summary_template = """
    你是一个专业的内容分析与重构专家。以下是一部小说多个章节的分块总结内容，请基于这些总结进行综合分析，并最终用简洁、有条理的小说语言重述整篇故事。

    请严格按照以下结构输出：

    1. 小说类型与风格：
    - 识别这部小说的主要类型（如奇幻、科幻、言情、悬疑等）。
    - 描述其整体风格（如黑暗、浪漫、讽刺、史诗感等）。

    2. 主要角色：
    - 列出关键人物及其核心特征（性格、动机、关系等），不要罗列次要人物。

    3. 关键情节要点：
    - 提炼出3-5个推动故事发展的核心事件或转折点。

    4. 情节重构（300-500字）：
    - 根据章节总结，写一篇连贯、完整的情节描述。
    - 必须严格依据原文内容，不能添加任何想象或修饰性语句（如"扣人心弦""人物形象丰满"等无实质内容的套话）。
    - 使用小说式语言，语言流畅自然，突出人物动机与冲突，展现情节逻辑。

    ---
    章节总结如下：
    {content}
    """

    summary_prompt = PromptTemplate(
        input_variables=["content"],
        template=summary_template
    )

    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        verbose=verbose
    )

    return summary_chain

def summarize_long_text(content, llm=ChatOpenAI(base_url="https://api.chatanywhere.tech/v1"), verbose=True):
    # 1 对长文本进行分词切割
    text_splitter = TokenTextSplitter(
        chunk_size=3500,
        chunk_overlap=300,
        encoding_name="gpt2"
    )
    docs = text_splitter.create_documents([content])

    # 2 对每个文档块进行总结
    chunk_summaries = []
    summary_chain = init_summary_chain(llm, verbose=verbose)
    
    for i, doc in enumerate(docs):
        print(f"正在总结第 {i+1}/{len(docs)} 个文档块...")
        summary = summary_chain.run(content=doc.page_content)
        chunk_summaries.append(summary)
        if verbose:
            print(summary)
            print(f"第 {i+1}/{len(docs)} 个文档块总结完成")
            print("-" * 50)

    # 3 将所有块的总结合并为一个长文本并进行最终总结
    all_summaries = "\n\n".join([f"文档块 {i+1} 总结:\n{summary}" 
                               for i, summary in enumerate(chunk_summaries)])
    
    longtext_summary_chain = init_longtext_summary_chain(llm, verbose=verbose)
    final_summary = longtext_summary_chain.run(content=all_summaries)
    return final_summary

# 初始化LLM
llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")

# 为主控Agent创建工具集
tools = [
    Tool(
        name="爬取B站视频",
        func=spider_bilibili,
        description="当URL包含'bilibili.com'时使用此工具爬取B站视频。输入参数为完整URL。"
    ),
    Tool(
        name="爬取笔趣阁小说",
        func=spider_biqugen,
        description="当URL包含'bie5'时使用此工具爬取笔趣阁小说。输入参数为完整URL。该工具会将内容保存到文件并返回文件路径。"
    ),
    Tool(
        name="总结文件内容",
        func=summarize_file_content,
        description="总结文本文件内容。如果不提供文件路径，会尝试使用上次爬取的文件。输入参数为文件路径(可选)。"
    ),
    Tool(
        name="创建向量数据库",
        func=setup_rag_database,
        description="将文本文件内容存入向量数据库，以便后续进行问答。如果不提供文件路径，会尝试使用上次爬取的文件。输入参数为文件路径(可选)。"
    ),
    Tool(
        name="查询内容",
        func=query_rag_database,
        description="在向量数据库中查询信息。如果不提供数据库路径，会尝试使用上次创建的数据库。输入参数为查询内容，可选数据库路径。"
    )
]

# 创建带记忆的Agent
memory = ConversationBufferMemory(memory_key="chat_history")

# 定义Agent提示模板
prefix = """你是一个内容分析助手，可以帮助用户爬取网页内容，总结文本，并通过向量数据库进行问答。
你有以下工具可以使用:"""

suffix = """请始终使用上述工具完成用户请求。在执行操作前，请根据用户输入决定需要使用的工具和执行顺序。

1. 对于包含URL的请求，先分析URL类型，选择合适的爬虫工具
2. 爬取内容后，可以选择总结内容或创建向量数据库
3. 如果用户想问问题，需要先确保已创建向量数据库

按照智能工作流程自动判断下一步操作。

当前系统状态:
- 上次爬取文件: {last_file}
- 上次创建数据库: {last_db}
- 处理历史: {history}

请根据用户的问题和当前状态，决定应该执行什么操作。

对话历史:
{chat_history}

用户问题: {input}
{agent_scratchpad}"""

# 格式化系统状态, 只提取最近的5条处理历史
def format_history():
    if not app_state.processing_history:
        return "无处理历史"
    return "\n".join([f"- {h['action']}: {h['details']}" for h in app_state.processing_history[-5:]])

# 创建Agent
agent_prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad", "last_file", "last_db", "history"]
)

llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

# 主函数
def process_user_request(user_input):
    return agent_executor.run(
        input=user_input,
        last_file=app_state.last_file_path or "无",
        last_db=app_state.last_db_path or "无",
        history=format_history()
    )

# 示例运行
if __name__ == "__main__":
    while True:
        user_input = input("\n请输入您的请求 (输入'退出'结束): ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            break
            
        try:
            response = process_user_request(user_input)
            print("\n助手回复:", response)
        except Exception as e:
            print(f"处理请求时出错: {str(e)}")