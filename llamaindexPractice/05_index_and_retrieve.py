"""
05_index_and_retrieve.py — 索引构建与检索

这一课学习 LlamaIndex 的核心：
1. 索引类型：VectorStoreIndex, SummaryIndex, KeywordTableIndex
2. 如何构建索引
3. 如何检索相关文档
4. 检索质量评估

【Java 程序员速查】
  索引 = 数据库的索引（加速查询）
  VectorStoreIndex = 向量索引（语义搜索）
  SummaryIndex = 摘要索引（全文搜索）
  KeywordTableIndex = 关键词索引（精确匹配）
  类比 Java:
    // Lucene 的索引
    IndexWriter writer = new IndexWriter(directory, config);
    writer.addDocument(document);
    IndexSearcher searcher = new IndexSearcher(reader);
    TopDocs results = searcher.search(query, 10);
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    SummaryIndex,
    KeywordTableIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import (
    VectorIndexRetriever,   # 向量检索器
    SummaryIndexRetriever,  # 摘要检索器
    KeywordTableSimpleRetriever,  # 关键词检索器
)

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
print("  索引构建与检索 — Index & Retrieve")
print("=" * 60)


# ============================================================
# 1. 什么是索引？
# ============================================================

def explain_index():
    """
    索引是 LlamaIndex 的核心抽象

    索引的作用：
    1. 组织文档数据
    2. 加速检索
    3. 支持多种检索策略

    类比 Java:
      // Lucene 的索引
      IndexWriter writer = new IndexWriter(directory, config);
      writer.addDocument(doc);
      writer.commit();

      // LlamaIndex 的索引
      index = VectorStoreIndex.from_documents(documents)
      // 底层自动：分块 → Embedding → 存储向量 → 建索引
    """
    print("\n>>> 什么是索引？")
    print("""
    索引 = 文档的组织方式 + 检索加速结构

    类比数据库的索引：
    - 没有索引：全表扫描（慢）
    - 有索引：B-Tree / Hash / 向量索引（快）

    LlamaIndex 的索引类型：
    - VectorStoreIndex: 向量索引（语义搜索）← 最常用
    - SummaryIndex: 全文索引（关键词搜索）
    - KeywordTableIndex: 关键词表索引
    - TreeIndex: 树形索引（层次化检索）
    """)


# ============================================================
# 2. VectorStoreIndex — 向量索引（最常用）
# ============================================================

def demo_vector_store_index():
    """
    VectorStoreIndex 是 LlamaIndex 最常用的索引类型

    工作原理：
    1. 对每个文档块生成 Embedding（向量）
    2. 将向量存入向量存储（ChromaDB / FAISS / Milvus）
    3. 查询时，对查询文本也生成向量
    4. 计算查询向量与文档向量的相似度
    5. 返回最相似的 Top-K 个文档块

    类比 Java:
      // 1. 创建索引
      VectorStoreIndex index = new VectorStoreIndex(documents);

      // 2. 创建检索器
      VectorIndexRetriever retriever = index.asRetriever();
      retriever.similarityTopK = 5;

      // 3. 检索
      List<NodeWithScore> nodes = retriever.retrieve("你的问题");
      for (NodeWithScore node : nodes) {
          System.out.println(node.getNode().getText());
          System.out.println("相似度: " + node.getScore());
      }
    """
    print("\n>>> 使用 VectorStoreIndex")

    # --- 【Python 语法】创建测试文档 ---
    # Document 是 LlamaIndex 的基本文档单元
    documents = [
        Document(
            text="LlamaIndex 是一个强大的框架，用于构建基于 LLM 的应用。"
                 "它支持文档加载、文本分块、嵌入、索引、检索和查询。",
            metadata={"source": "llamaindex_docs", "chapter": "intro"},
        ),
        Document(
            text="RAG（检索增强生成）是一种结合检索和生成的技术。"
                 "它先用向量数据库检索相关文档，再将文档作为上下文提供给 LLM。",
            metadata={"source": "rag_docs", "chapter": "rag"},
        ),
        Document(
            text="向量数据库（如 Milvus、ChromaDB、FAISS）专门用于存储和检索高维向量。"
                 "它们支持相似度搜索，是 RAG 系统的核心组件。",
            metadata={"source": "vector_db_docs", "chapter": "vectors"},
        ),
        Document(
            text="Embedding 模型将文本转换为固定维度的向量。"
                 "OpenAI 的 text-embedding-ada-002 生成 1536 维向量。",
            metadata={"source": "embedding_docs", "chapter": "embeddings"},
        ),
        Document(
            text="文本分块（Chunking）是将长文档切分为小块的过程。"
                 "合适的分块策略可以提升检索质量和回答准确性。",
            metadata={"source": "chunking_docs", "chapter": "chunking"},
        ),
    ]

    # --- 【Python 语法】SentenceSplitter ---
    # 对文档进行分块
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    nodes = splitter.split_documents(documents)
    print(f"  文档数量: {len(documents)}")
    print(f"  分块数量: {len(nodes)}")

    # --- 【Python 语法】VectorStoreIndex.from_documents() ---
    # 从文档列表一步构建向量索引
    # 内部自动完成：分块 → Embedding → 存储向量
    # 类比 Java: VectorStoreIndex index = VectorStoreIndex.fromDocuments(documents);
    index = VectorStoreIndex.from_documents(documents)
    print("  ✓ 向量索引构建完成")

    # --- 【Python 语法】.as_retriever() ---
    # 将索引转换为检索器
    # 参数：
    #   similarity_top_k: 返回最相似的 Top-K 个节点（默认 2）
    #   similarity_cutoff: 相似度阈值（低于此值的节点不返回）
    # 类比 Java: index.asRetriever(5, 0.5);
    retriever = index.as_retriever(similarity_top_k=3)
    print("  ✓ 检索器已创建")

    # --- 【Python 语法】.retrieve() ---
    # 执行检索，返回相关节点列表
    # 每个节点包含：
    #   - node.text: 节点文本
    #   - node.score: 相似度分数（0-1，越高越相似）
    #   - node.metadata: 元数据
    # 类比 Java: retriever.retrieve("你的问题");
    query = "什么是 RAG？"
    print(f"\n  查询: {query}")
    results = retriever.retrieve(query)

    print(f"\n  检索结果（Top-{len(results)}）:")
    for i, result in enumerate(results):
        print(f"    #{i+1} 相似度: {result.score:.4f}")
        print(f"       文本: {result.text[:80]}...")
        print(f"       来源: {result.metadata.get('source', 'N/A')}")


# ============================================================
# 3. SummaryIndex — 摘要索引
# ============================================================

def demo_summary_index():
    """
    SummaryIndex 将所有文档内容合并为一个长文本

    工作原理：
    1. 将所有文档块拼接成一个长字符串
    2. 查询时，将整个文本发送给 LLM
    3. LLM 基于全文生成回答

    适用场景：
    - 小文档（总 token < LLM 上下文窗口）
    - 需要全局理解的查询

    不适用场景：
    - 大文档（超出上下文窗口）
    - 需要精确定位的查询

    类比 Java:
      // 将所有文档拼接
      String fullText = documents.stream()
          .map(Document::getText)
          .collect(Collectors.joining("\n\n"));

      // 发送给 LLM
      String response = llm.generate("基于以下文档回答: " + fullText);
    """
    print("\n>>> 使用 SummaryIndex")

    documents = [
        Document(text="Python 是一种解释型、面向对象的编程语言。"),
        Document(text="Java 是一种编译型、面向对象的编程语言。"),
        Document(text="JavaScript 是一种主要用于 Web 开发的脚本语言。"),
    ]

    # --- 【Python 语法】SummaryIndex.from_documents() ---
    # 从文档构建摘要索引
    index = SummaryIndex.from_documents(documents)
    print("  ✓ 摘要索引构建完成")

    # --- 【Python 语法】.as_retriever() ---
    # 摘要索引的检索器返回所有节点
    retriever = index.as_retriever()
    query = "比较这些编程语言的特点"
    print(f"\n  查询: {query}")
    results = retriever.retrieve(query)

    print(f"\n  检索结果（共 {len(results)} 个节点）:")
    for i, result in enumerate(results):
        print(f"    #{i+1}: {result.text}")


# ============================================================
# 4. KeywordTableIndex — 关键词索引
# ============================================================

def demo_keyword_table_index():
    """
    KeywordTableIndex 提取文档中的关键词并建立索引

    工作原理：
    1. 从文档中提取关键词（基于 TF-IDF 或 BM25）
    2. 建立关键词 → 文档的映射
    3. 查询时，提取查询关键词，查找匹配的文档

    适用场景：
    - 需要精确关键词匹配的查询
    - 与向量搜索互补（混合搜索）

    类比 Java:
      // 建立倒排索引
      Map<String, List<Document>> keywordIndex = new HashMap<>();
      for (Document doc : documents) {
          List<String> keywords = extractKeywords(doc.getText());
          for (String keyword : keywords) {
              keywordIndex.computeIfAbsent(keyword, k -> new ArrayList<>()).add(doc);
          }
      }
    """
    print("\n>>> 使用 KeywordTableIndex")

    documents = [
        Document(text="Python 支持多种编程范式：面向对象、函数式、命令式。"),
        Document(text="Java 是一种强类型的面向对象编程语言。"),
        Document(text="TypeScript 是 JavaScript 的超集，增加了类型系统。"),
    ]

    # --- 【Python 语法】KeywordTableIndex.from_documents() ---
    # 从文档构建关键词索引
    index = KeywordTableIndex.from_documents(documents)
    print("  ✓ 关键词索引构建完成")

    # --- 【Python 语法】.as_retriever() ---
    retriever = index.as_retriever()
    query = "Python 编程语言"
    print(f"\n  查询: {query}")
    results = retriever.retrieve(query)

    print(f"\n  检索结果（共 {len(results)} 个节点）:")
    for i, result in enumerate(results):
        print(f"    #{i+1}: {result.text}")


# ============================================================
# 5. 检索质量评估
# ============================================================

def evaluate_retrieval():
    """
    评估检索质量的方法

    1. 人工评估：查看检索结果是否相关
    2. 自动评估：
       - 相似度分数：越高越相关
       - 召回率：是否找到了所有相关文档
       - 精确率：返回的文档中有多少是相关的

    调优建议：
    - similarity_top_k: 增加 → 召回率高但可能混入噪声
    - similarity_cutoff: 提高 → 只返回高相似度结果
    - chunk_size: 太小 → 丢失上下文；太大 → 引入噪声
    """
    print("\n" + "=" * 60)
    print("  检索质量评估")
    print("=" * 60)
    print("""
    评估方法：
    1. 人工查看检索结果的相关性
    2. 观察相似度分数分布
    3. 对比不同参数的效果

    调优参数：
    - similarity_top_k: 返回数量（默认 2）
    - similarity_cutoff: 相似度阈值（默认 0.7）
    - chunk_size: 分块大小（默认 1024）
    - chunk_overlap: 分块重叠（默认 200）

    经验法则：
    - 中文文本：chunk_size 设为 512-1024
    - 英文文本：chunk_size 设为 768-1536
    - chunk_overlap 设为 chunk_size 的 10-20%
    """)


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数：依次演示各种索引类型
    """
    explain_index()
    demo_vector_store_index()
    demo_summary_index()
    demo_keyword_table_index()
    evaluate_retrieval()

    print("\n" + "=" * 60)
    print("  OK 索引构建与检索完成！")
    print("=" * 60)
    print("""
下一步：
  - 理解不同索引类型的适用场景
  - VectorStoreIndex 是最常用的（语义搜索）
  - 下一课学习查询引擎（Query Engine）
    """)


if __name__ == "__main__":
    main()
