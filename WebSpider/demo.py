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

llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")

def spider_bilibili(url):
    bilibili_main(url)
    return "bilibili视频爬取完毕"

def spider_biqugen(url):
    biqugen_main(url)
    return "笔趣阁小说爬取完毕"

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
    verbose=True
)

# 创建内容分析和总结的提示模板
summary_template = """
你是一个专业的内容分析师。请分析以下内容，确定其类型（如视频内容、小说文本等），
然后提供一个详尽的总结。如果内容过长，请提取关键信息。

内容：
{content}

请提供以下分析：
1. 内容类型：这是什么类型的内容？
2. 主要主题：内容主要讲述了什么？
3. 关键要点：内容中最重要的3-5个要点是什么？
4. 总结：用300字左右总结内容的精华。
"""

# 创建总结链
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template=summary_template
)

summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    verbose=True
)

if __name__ == "__main__":
    res = spider_agent.run("你好啊，你都会些什么？")
    print(res)

    # 示例 1: 爬取 Bilibili 视频
    bilibili_url = "https://www.bilibili.com/video/BV1GYGtzmEEN"
    result_1 = spider_agent.run(f"请爬取以下url: {bilibili_url}")
    print(result_1)

    # 示例 2: 爬取笔趣阁小说
    biqugen_url = "https://www.bie5.cc/html/45771/"
    result_2 = spider_agent.run(f"请爬取以下url: {biqugen_url}")
    print(result_2)