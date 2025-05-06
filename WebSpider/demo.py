import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from WebSpider_bilibili import main as bilibili_main
from WebSpider_biqugen import main as biqugen_main

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")

functions = [
    {
        'name': 'spider_bilibili',
        'description': '通过提供url爬取bilibili视频',
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'url链接',
                },
            },
            'required': ['url'],
        },
    },
    {
        'name': 'spider_biqugen',
        'description': '通过提供url爬取笔趣阁小说',
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'url链接',
                },
            },
            'required': ['url'],
        },
    }
]

def spider_bilibili(url):
    bilibili_main(url)
    return "bilibili视频爬取完毕"

def spider_biqugen(url):
    biqugen_main(url)
    return "笔趣阁小说爬取完毕"

# 定义工具
tools = [
    Tool(
        name="spider_bilibili",
        func=spider_bilibili,
        description="通过提供url爬取bilibili视频"
    ),
    Tool(
        name="spider_biqugen",
        func=spider_biqugen,
        description="通过提供url爬取笔趣阁小说"
    )
]

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == "__main__":
    # # 示例 1: 爬取 Bilibili 视频
    # bilibili_url = "https://www.bilibili.com/video/BV1GYGtzmEEN"
    # print("调用 Bilibili 爬取工具:")
    # result_1 = agent.run(f"请爬取以下 Bilibili 视频: {bilibili_url}")
    # print(result_1)
 
    # 示例 2: 爬取笔趣阁小说
    biqugen_url = "https://www.bie5.cc/html/45771/"
    print("\n调用笔趣阁爬取工具:")
    result_2 = agent.run(f"请爬取以下笔趣阁小说: {biqugen_url}")
    print(result_2)