pip install chainlit报错
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
opentelemetry-instrumentation-asgi 0.53b1 requires opentelemetry-instrumentation==0.53b1, but you have opentelemetry-instrumentation 0.52b1 which is incompatible.
opentelemetry-instrumentation-asgi 0.53b1 requires opentelemetry-semantic-conventions==0.53b1, but you have opentelemetry-semantic-conventions 0.52b1 which is incompatible.
opentelemetry-instrumentation-fastapi 0.53b1 requires opentelemetry-instrumentation==0.53b1, but you have opentelemetry-instrumentation 0.52b1 which is incompatible.
opentelemetry-instrumentation-fastapi 0.53b1 requires opentelemetry-semantic-conventions==0.53b1, but you have opentelemetry-semantic-conventions 0.52b1 which is incompatible.
```

任务清单：
- [x] 利用RAG，将给定的PDF加载、切割、索引、检索，将其转换为大模型的知识库。然后使用大模型进行问答。
- [x] 实现上述功能后，将其融入到爬虫程序中，让其总结爬取的小说内容（最好是多给几章）。
- [ ] 尝试使用Embedding将小说内容转换为向量，实现QA功能。
- [ ] 尝试寻找总结视频内容的方式，没有的话可以换其他内容，但不能是文字。
- [ ] 同样的，若上述完成，将其融入到爬虫程序中。