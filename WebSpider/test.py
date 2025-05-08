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
    你是一个专业的内容分析与重构专家。以下是一部小说多个章节的分块总结内容，请基于这些总结进行综合分析，并最终用简洁、有条理的小说语言重述整篇故事。

    请严格按照以下结构输出：

    1. 小说类型与风格：
    - 识别这部小说的主要类型（如奇幻、科幻、言情、悬疑等）。
    - 描述其整体风格（如黑暗、浪漫、讽刺、史诗感等）。

    2. 主要角色：
    - 列出关键人物及其核心特征（性格、动机、关系等），不要罗列次要人物。

    3. 关键情节要点：
    - 提炼出3-5个推动故事发展的核心事件或转折点。

    4. 情节重构（800-1000字）：
    - 根据章节总结，写一篇连贯、完整的情节描述。
    - 必须严格依据原文内容，不能添加任何想象或修饰性语句（如“扣人心弦”“人物形象丰满”等无实质内容的套话）。
    - 使用小说式语言，语言流畅自然，突出人物动机与冲突，展现情节逻辑。

    ---
    章节总结如下：
    {content}
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