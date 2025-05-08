import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./api_key.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import TokenTextSplitter
import tiktoken
from langchain.chains.summarize import load_summarize_chain

llm = ChatOpenAI(base_url="https://api.chatanywhere.tech/v1")

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
4. 总结：用200字左右总结内容的精华。
"""

# 创建总结链
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template=summary_template
)

summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    verbose=False
)

# chain = load_summarize_chain(
#     llm=llm,
#     chain_type="refine",
#     verbose=True,
# )

file_path = ["novel\斗破苍穹\第一章 陨落的天才.txt", "novel\斗破苍穹\第二章 斗气大陆.txt", "novel\斗破苍穹\第三章 客人.txt", "novel\斗破苍穹\第四章 云岚宗.txt"]
content = ""

for path in file_path:
    with open(path, "r", encoding="utf-8") as f:
        content += f.read()

text_spliter = TokenTextSplitter(
    chunk_size=3500,
    chunk_overlap=300,
    encoding_name="gpt2"
)

docs = text_spliter.create_documents([content])

# print(chain.run(docs))
# print(summary_chain.run(content=content))

def summarize_long_text(docs, summary_chain):
    # 1 对每个文档块进行总结
    chunk_summaries = []

    for i, doc in enumerate(docs):
        print(f"正在总结第 {i+1}/{len(docs)} 个文档块...")
        summary = summary_chain.run(content=doc.page_content)
        chunk_summaries.append(summary)
        print(summary)
        print(f"第 {i+1}/{len(docs)} 个文档块总结完成")
        print("-" * 50)

    # 2 将所有块的总结合并为一个长文本
    all_summaries = "\n\n".join([f"文档块 {i+1} 总结:\n{summary}" 
                               for i, summary in enumerate(chunk_summaries)])

    # 创建元总结模板
    meta_summary_template = """
    你是一个专业的内容分析与综合专家。下面是一部小说多个部分的分块总结。
    请根据这些总结内容，提供一个全面的分析与总结。
    
    章节总结:
    {content}
    
    请提供以下整体分析:
    1. 小说类型与风格: 这是什么类型的小说？
    2. 主要情节: 故事主要讲述了什么？
    3. 主要角色: 出现了哪些重要角色及其特点？
    4. 关键要点: 故事中最重要的3-5个要点是什么？
    5. 总体评价: 这几章内容展示了怎样的主题和风格？
    6. 最终总结: 用800-1000字总结这几章的内容。
    """

    meta_summary_prompt = PromptTemplate(
        input_variables=["content"],
        template=meta_summary_template
    )

    meta_summary_chain = LLMChain(
        llm=llm,
        prompt=meta_summary_prompt,
        verbose=True
    )
    
    # 生成最终总结
    final_summary = meta_summary_chain.run(content=all_summaries)
    
    return final_summary

final_summary = summarize_long_text(docs, summary_chain)
print("\n=== 最终总结 ===\n")
print(final_summary)