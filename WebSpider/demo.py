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

def spider_bilibili(url):
    bilibili_main(url)
    return "bilibili视频爬取完毕"

def spider_biqugen(url):
    biqugen_main(url)
    return "笔趣阁小说爬取完毕"

def init_spider_agent(llm, verbose=True):
    """
    初始化 Agent
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
        verbose=verbose
    )
    return spider_agent

if __name__ == "__main__":
    llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")

    spider_agent = initialize_agent(llm)
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