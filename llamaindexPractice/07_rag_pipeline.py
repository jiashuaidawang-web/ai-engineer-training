"""
07_rag_pipeline.py — 完整 RAG 管道

这一课将所有知识点整合，构建一个完整的 RAG 系统：
1. 文档加载 → 2. 文本分块 → 3. Embedding → 4. 向量存储 → 5. 检索 → 6. LLM 生成

你将学会：
- 端到端构建 RAG 系统
- 调优各阶段的参数
- 评估回答质量
- 处理常见问题

【Java 程序员速查】
  RAG = Retrieval-Augmented Generation（检索增强生成）
  类比 Spring Boot 的架构：
    Controller（查询引擎） → Service（RAG Pipeline） → Repository（向量存储）
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate

# 配置 LLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)
Settings.embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

print("=" * 60)
print("  完整 RAG 管道 — End-to-End RAG")
print("=" * 60)


# ============================================================
# RAG 架构总览
# ============================================================

def show_rag_architecture():
    """
    RAG 架构图解

    用户问题 → 向量化 → 向量检索 → 检索结果 → 组装 Prompt → LLM → 回答

    类比 Java 的 MVC 架构：
    ┌─────────────────────────────────────────────────────────┐
    │                    View (用户界面)                       │
    │              用户提问 / 显示回答                          │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │                 Controller (查询引擎)                     │
    │           接收问题 → 调用 Service                         │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │                   Service (RAG Pipeline)                  │
    │  1. 问题向量化                                             │
    │  2. 向量检索                                               │
    │  3. 组装 Prompt                                            │
    │  4. 调用 LLM 生成回答                                       │
    └──────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────┐
    │                Repository (向量存储)                      │
    │        存储和检索向量数据                                   │
    └─────────────────────────────────────────────────────────┘
    """
    print("\n>>> RAG 架构")
    print("""
    用户问题 → 向量化 → 向量检索 → 检索结果 → Prompt → LLM → 回答
    """)


# ============================================================
# 第一步：构建 RAG 系统
# ============================================================

def build_rag_system():
    """
    构建完整的 RAG 系统

    这是 RAG 的核心流程，每一步都很关键：
    1. 加载文档
    2. 文本分块
    3. 构建索引
    4. 创建查询引擎
    """
    print("\n>>> 构建 RAG 系统")

    # --- 1. 加载文档 ---
    # 从本地目录加载文档
    data_dir = os.path.join(os.path.dirname(__file__), "..", "llamaindex-demo", "data")

    if os.path.exists(data_dir):
        print("  [1/4] 加载文档...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"       加载了 {len(documents)} 个文档")
    else:
        print("  [1/4] 使用示例文档（数据目录不存在）")
        documents = [
            Document(text="LlamaIndex 是一个用于构建 LLM 应用的框架。"
                          "它支持文档加载、文本分块、嵌入、索引和查询。"),
            Document(text="RAG（检索增强生成）结合了信息检索和文本生成的优势。"
                          "先用向量数据库检索相关文档，再将文档作为上下文提供给 LLM。"),
            Document(text="向量数据库专门用于存储和检索高维向量。"
                          "它们支持高效的相似度搜索，是 RAG 系统的核心组件。"),
            Document(text="文本分块是将长文档切分为小块的过程。"
                          "合适的分块策略可以提升检索质量和回答准确性。"),
            Document(text="Embedding 模型将文本转换为固定维度的向量。"
                          "OpenAI 的 text-embedding-ada-002 生成 1536 维向量。"),
        ]

    # --- 2. 文本分块 ---
    print("  [2/4] 文本分块...")
    # --- 【Python 语法】SentenceSplitter ---
    # chunk_size: 每个块的最大 token 数
    # chunk_overlap: 相邻块的重叠 token 数
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.split_documents(documents)
    print(f"       切分为 {len(nodes)} 个块")

    # --- 3. 构建索引 ---
    print("  [3/4] 构建向量索引...")
    # --- 【Python 语法】VectorStoreIndex.from_documents() ---
    # 一步完成：Embedding + 存储向量 + 建索引
    index = VectorStoreIndex.from_documents(nodes)
    print("       ✓ 索引构建完成")

    # --- 4. 创建查询引擎 ---
    print("  [4/4] 创建查询引擎...")
    # --- 【Python 语法】自定义检索器和响应合成器 ---
    retriever = index.as_retriever(
        similarity_top_k=3,    # 返回最相似的 3 个节点
        similarity_cutoff=0.5, # 相似度阈值
    )

    # --- 【Python 语法】CompactAndRefine ---
    # 响应合成策略：先压缩文档，再逐步精炼回答
    synthesizer = CompactAndRefine(verbose=True)

    # --- 【Python 语法】RetrieverQueryEngine.from_defaults() ---
    query_engine = RetrieverQueryEngine.from_defaults(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )
    print("       ✓ 查询引擎已创建")

    return query_engine


# ============================================================
# 第二步：测试 RAG 系统
# ============================================================

def test_rag_system(query_engine):
    """
    测试 RAG 系统的回答质量

    测试不同难度的问题：
    1. 简单事实查询
    2. 归纳总结
    3. 复杂推理
    """
    print("\n" + "=" * 60)
    print("  测试 RAG 系统")
    print("=" * 60)

    # --- 【Python 语法】测试问题列表 ---
    questions = [
        # 简单事实查询
        "LlamaIndex 是什么？",
        "什么是 RAG？",
        # 归纳总结
        "向量数据库在 RAG 中起什么作用？",
        # 复杂推理
        "如何优化 RAG 系统的检索质量？",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 问题: {question}")
        print("-" * 50)

        # --- 【Python 语法】query_engine.query() ---
        # 发送查询，获取回答
        response = query_engine.query(question)

        # --- 【Python 语法】response.response ---
        # 获取 LLM 的回答
        print(f"回答: {response.response}")

        # --- 【Python 语法】response.source_nodes ---
        # 获取检索到的相关节点（用于追溯）
        if hasattr(response, 'source_nodes'):
            print(f"\n参考文档（{len(response.source_nodes)} 个）:")
            for j, node in enumerate(response.source_nodes):
                print(f"  文档 {j+1} 相似度: {node.score:.4f}")
                print(f"    内容: {node.text[:100]}...")


# ============================================================
# 第三步：调优 RAG 系统
# ============================================================

def optimize_rag():
    """
    RAG 系统调优指南

    影响 RAG 质量的三个关键因素：
    1. 文档质量：干净的文档 → 更好的嵌入
    2. 分块策略：合适的 chunk_size 和 overlap
    3. 检索参数：similarity_top_k 和 cutoff

    调优建议：
    - 增加 similarity_top_k → 召回率高但可能混入噪声
    - 提高 similarity_cutoff → 只返回高相似度结果
    - 增大 chunk_size → 保留更多上下文但可能引入噪声
    - 使用重排序（Reranking）→ 提升检索质量
    """
    print("\n" + "=" * 60)
    print("  RAG 系统调优")
    print("=" * 60)
    print("""
    调优参数：
    1. chunk_size: 512-1024（中文）/ 768-1536（英文）
    2. chunk_overlap: chunk_size 的 10-20%
    3. similarity_top_k: 3-10（根据文档数量调整）
    4. similarity_cutoff: 0.5-0.8（过滤低相似度结果）

    进阶优化：
    1. 重排序（Reranking）：用交叉编码器对检索结果重新排序
    2. 混合搜索：向量搜索 + 关键词搜索
    3. 元数据过滤：按来源/日期/类别过滤
    4. 查询改写：将用户问题改写为更适合检索的形式
    """)


# ============================================================
# 第四步：常见问题排查
# ============================================================

def troubleshoot():
    """
    常见问题及解决方案

    1. 回答不准确
       - 检查文档是否包含了相关信息
       - 增加 similarity_top_k
       - 调整 chunk_size

    2. 回答太慢
       - 减少检索的文档数量
       - 使用更快的 Embedding 模型
       - 考虑缓存热门查询

    3. 回答幻觉
       - 提高 similarity_cutoff
       - 使用更严格的提示模板
       - 添加"如果找不到答案，请说不知道"的指令

    4. 文档加载失败
       - 检查文件格式是否支持
       - 确认文件编码（UTF-8）
       - 查看错误日志
    """
    print("\n" + "=" * 60)
    print("  常见问题排查")
    print("=" * 60)
    print("""
    1. 回答不准确 → 增加 similarity_top_k，调整 chunk_size
    2. 回答太慢 → 减少检索数量，使用缓存
    3. 回答幻觉 → 提高 similarity_cutoff，严格提示
    4. 文档加载失败 → 检查格式和编码
    """)


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数：构建并测试 RAG 系统
    """
    show_rag_architecture()
    query_engine = build_rag_system()
    test_rag_system(query_engine)
    optimize_rag()
    troubleshoot()

    print("\n" + "=" * 60)
    print("  OK 完整 RAG 管道完成！")
    print("=" * 60)
    print("""
恭喜！你已经完成了 LlamaIndex 的核心学习路径。

下一步（生产环境）：
1. 接入真实文档（PDF, Word, Excel 等）
2. 使用本地 Embedding 模型（无需 API Key）
3. 添加重排序（Reranking）提升检索质量
4. 实现缓存机制加速热门查询
5. 部署为 API 服务（FastAPI）
    """)


if __name__ == "__main__":
    main()
