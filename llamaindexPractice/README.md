# LlamaIndex 学习路径 — 从零到生产级 RAG

> 目标：通过手写代码，彻底掌握 LlamaIndex，最终能独立构建生产级 RAG 系统。
> 面向 Java 程序员，零基础 Python。

## 学习路线（共 8 课）

| 课 | 文件 | 内容 | 预计时间 |
|----|------|------|----------|
| 0 | `00_python_syntax_cheatsheet.py` | Python 语法速查（Java 程序员必看） | 20min |
| 1 | `01_llamaindex_hello.py` | LlamaIndex 入门：安装、配置、第一个 Demo | 20min |
| 2 | `02_document_ingestion.py` | 文档加载：PDF/Markdown/TXT 解析 | 30min |
| 3 | `03_text_splitting.py` | 文本分块策略：Recursive/Token/Sentence | 30min |
| 4 | `04_embedding_and_storage.py` | Embedding + 向量存储（Chroma/FAISS） | 30min |
| 5 | `05_index_and_retrieve.py` | 索引构建 + 检索（VectorIndex/SummaryIndex） | 30min |
| 6 | `06_query_engine.py` | 查询引擎：Simple、Retriever、Chat | 30min |
| 7 | `07_rag_pipeline.py` | 完整 RAG 管道：检索 → 重排 → 生成 | 40min |
| 8 | `08_production_rag.py` | 生产级 RAG：混合搜索、工作流、缓存 | 40min |

## 前置条件

1. **Python 3.9+**（你已安装）
2. **OpenAI API Key**（已有，配置在 `.env`）
3. **pip 包管理**

## 安装步骤

```bash
# 安装核心包 + 常用插件
pip install llama-index llama-index-core llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-file chromadb

# 如果需要本地向量库（不依赖 OpenAI）
pip install llama-index-vector-stores-faiss

# 如果需要本地 Embedding
pip install sentence-transformers
```

## 运行方式

```bash
# 确保 .env 中有 OPENAI_API_KEY
# 按顺序运行每个 .py 文件
python 00_python_syntax_cheatsheet.py
python 01_llamaindex_hello.py
# ...
```

## 配套文档

- `LLAMAINDEX_CORE_CONCEPTS.md` — LlamaIndex 核心概念总结（API速查/索引对比/调优建议/排错）

## 学习建议

1. **先跑通，再理解**：每个文件先运行看输出，再读注释
2. **改参数**：把 chunk_size 改小/改大，观察结果变化
3. **对照文档**：遇到不懂的 API，去 [LlamaIndex 文档](https://docs.llamaindex.ai/) 查
4. **从简到繁**：严格按 0→7 的顺序，后面的依赖前面的概念
