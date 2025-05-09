import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from WebSpider_bilibili import main as bilibili_main
from WebSpider_biqugen import main as biqugen_main

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def spider_bilibili(url):
    res = bilibili_main(url)
    return res

def spider_biqugen(url):
    contents = biqugen_main(url, first=15)

    # 生成唯一文件名并保存内容
    import time
    file_path = f"./novel/content_{int(time.time())}.txt"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(contents)
    
    # 返回文件路径和简短信息，而不是返回大量文本
    return f"内容已保存到文件: {file_path}，总字符数: {len(contents)}"

def init_spider_agent(llm, verbose=True):
    """初始化 Agent"""

    custom_prefix = """
    你是一个专业的网页内容爬取助手。你的任务是帮助用户爬取网页内容。
    当你成功爬取内容并保存到文件时，请在回答中包含完整的文件路径，不能有其他内容
    请确保文件路径使用完整格式，便于后续处理。
    """

    # 定义工具
    spider_tools = [
        Tool(
            name="spider_bilibili",
            func=spider_bilibili,
            description="用于爬取 Bilibili 视频。当 URL 中包含 'bilibili.com' 时使用此工具。"
        ),
        Tool(
            name="spider_biqugen",
            func=spider_biqugen,
            description="用于爬取笔趣阁小说。当 URL 中包含 'bie5' 时使用此工具。"
        )
    ]

    # 初始化 Agent
    spider_agent = initialize_agent(
        tools=spider_tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=verbose,
        agent_kwargs={"prefix": custom_prefix},
    )
    return spider_agent

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
    - 必须严格依据原文内容，不能添加任何想象或修饰性语句（如“扣人心弦”“人物形象丰满”等无实质内容的套话）。
    - 使用小说式语言，语言流畅自然，突出人物动机与冲突，展现情节逻辑。
    """

    # 创建总结链
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
    - 必须严格依据原文内容，不能添加任何想象或修饰性语句（如“扣人心弦”“人物形象丰满”等无实质内容的套话）。
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

def summarize_long_text(content, llm, verbose=True):
    # 1 对长文本进行分词切割
    text_spider = TokenTextSplitter(
        chunk_size=3500,
        chunk_overlap=300,
        encoding_name="gpt2"
    )
    docs = text_spider.create_documents([content])

    # 2 对每个文档块进行总结
    chunk_summaries = []
    summary_chain = init_summary_chain(llm, verbose=verbose)
    
    for i, doc in enumerate(docs):
        print(f"正在总结第 {i+1}/{len(docs)} 个文档块...")
        summary = summary_chain.run(content=doc.page_content)
        chunk_summaries.append(summary)
        print(summary)
        print(f"第 {i+1}/{len(docs)} 个文档块总结完成")
        print("-" * 50)

    # 3 将所有块的总结合并为一个长文本并进行最终总结
    # join() 是 Python 字符串的一个方法，用于将序列（如列表、元组）中的元素连接成一个单一的字符串。
    all_summaries = "\n\n".join([f"文档块 {i+1} 总结:\n{summary}" 
                               for i, summary in enumerate(chunk_summaries)])
    
    longtext_summary_chain = init_longtext_summary_chain(llm, verbose=verbose)
    final_summary = longtext_summary_chain.run(content=all_summaries)
    return final_summary

def summary_novel(url, llm, spider_agent):
    file_path = spider_agent.run(f"请爬取以下url: {url}")
    print(file_path)

    try:
        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        exit(1)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        exit(1)
    print(f"文件内容长度: {len(content)}")
    return summarize_long_text(content, llm)

def RAG(file_path):
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

if __name__ == "__main__":
    # llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")
    # spider_agent = init_spider_agent(llm)

    # res = spider_agent.run("你好啊，你都会些什么？")
    # print(res)

    # # 示例 1: 爬取 Bilibili 视频
    # bilibili_url = "https://www.bilibili.com/video/BV1GYGtzmEEN"/88888888888888888889888888888888 
    # result_1 = spider_agent.run(f"请爬取以下url: {bilibili_url}")
    # print(result_1)

    # # 示例 2: 爬取笔趣阁小说
    # biqugen_url = "https://www.bie5.cc/html/45771/"
    # res = summary_novel(biqugen_url, llm, spider_agent)
    # print(res)
    RAG("./novel/content_1746695216.txt")
