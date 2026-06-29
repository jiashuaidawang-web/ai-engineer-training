"""
04_embedding_and_storage.py — Embedding 与向量存储

这一课学习 LlamaIndex 的两个核心组件：
1. Embedding：将文本转换为向量
2. 向量存储：存储和检索向量

你将学会：
- 使用 OpenAI Embedding 模型
- 使用本地 Embedding 模型（sentence-transformers）
- 使用 ChromaDB 和 FAISS 作为向量存储
- 理解向量搜索的原理

【Java 程序员速查】
  Embedding = 文本 → 向量（高维数组）
  向量存储 = 数据库（专门存向量，支持相似度搜索）
  类比 Java:
    float[] vector = embeddingModel.encode(text);  // Embedding
    vectorStore.add(vector, metadata);              // 存储
    vectorStore.search(queryVector, k);             // 检索
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.schema import TextNode  # 文本节点
from llama_index.embeddings.openai import OpenAIEmbedding  # OpenAI Embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 本地 Embedding
from llama_index.core.vector_stores import (  # 向量存储模块
    SimpleVectorStore,    # 内存向量存储（简单，适合学习）
)

# 配置 LLM
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

print("=" * 60)
print("  Embedding 与向量存储")
print("=" * 60)


# ============================================================
# 1. 什么是 Embedding？
# ============================================================

def explain_embedding():
    """
    Embedding 是将文本转换为高维向量的过程

    原理：
    - 每个词/句子被映射为一个固定维度的向量
    - 语义相似的文本，向量之间的距离近
    - 向量空间中的几何关系反映语义关系

    举例：
    "猫" 和 "狗" 的向量距离近（都是宠物）
    "猫" 和 "汽车" 的向量距离远（完全不同领域）

    类比 Java:
      // 假设有一个 128 维的向量空间
      float[] catVector = {0.12, -0.34, 0.87, ...};  // 128 个浮点数
      float[] dogVector = {0.15, -0.31, 0.85, ...};  // 和 cat 很接近
      float[] carVector = {0.92, 0.11, -0.45, ...};  // 和 cat 很远

      // 计算相似度：余弦相似度
      double similarity = cosineSimilarity(catVector, dogVector);  // 接近 1
      double dissimilarity = cosineSimilarity(catVector, carVector);  // 接近 0
    """
    print("\n>>> 什么是 Embedding？")
    print("""
    Embedding = 文本 → 高维向量（数字数组）

    语义相似的文本，向量距离近
    语义不同的文本，向量距离远

    类比：在地图上看，城市 A 和城市 B 的距离近，
    说明它们地理位置相似。Embedding 把语义关系变成了空间距离。
    """)


# ============================================================
# 2. 使用 OpenAI Embedding
# ============================================================

def demo_openai_embedding():
    """
    使用 OpenAI 的 Embedding 模型

    OpenAI 提供两个 Embedding 模型：
    - text-embedding-ada-002: 1536 维，最常用
    - text-embedding-3-small: 1536/512/256 维可选，更快更便宜

    使用步骤：
    1. 创建 Embedding 对象
    2. 调用 embed_query() 对查询文本编码
    3. 调用 embed_documents() 对文档列表编码

    类比 Java:
      OpenAIEmbeddingModel model = OpenAIEmbeddingModel.builder()
          .apiKey(apiKey)
          .modelName("text-embedding-ada-002")
          .build();

      float[] queryVector = model.embed("你的问题");
      List<float[]> docVectors = model.embed(documents);
    """
    print("\n>>> 使用 OpenAI Embedding")

    # --- 【Python 语法】OpenAIEmbedding ---
    # 创建 OpenAI Embedding 实例
    embed_model = OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # --- 【Python 语法】embed_query() ---
    # 对单个查询文本进行编码，返回向量（list[float]）
    # 类比 Java: model.embed("你的问题")
    query = "什么是向量数据库？"
    query_vector = embed_model.get_text_embedding(query)
    print(f"\n查询: {query}")
    print(f"向量维度: {len(query_vector)}")
    print(f"向量前 10 个值: {query_vector[:10]}")

    # --- 【Python 语法】embed_documents() ---
    # 对文档列表进行编码，返回向量列表（list[list[float]]）
    # 类比 Java: model.embed(documents)
    documents = [
        "向量数据库是一种专门存储和搜索向量数据的数据库。",
        "机器学习模型可以将文本转换为高维向量。",
        "余弦相似度用于衡量两个向量的相似程度。",
    ]
    doc_vectors = [embed_model.get_text_embedding(doc) for doc in documents]

    print(f"\n文档数量: {len(documents)}")
    for i, (doc, vec) in enumerate(zip(documents, doc_vectors)):
        print(f"  文档 {i+1} 向量维度: {len(vec)}")
        print(f"    内容: {doc[:30]}...")

    return query_vector, doc_vectors


# ============================================================
# 3. 使用本地 Embedding（无需 API Key）
# ============================================================

def demo_local_embedding():
    """
    使用 HuggingFace 的本地 Embedding 模型

    优点：
    - 不需要 API Key
    - 数据不出本地，隐私性好
    - 免费

    缺点：
    - 速度较慢（CPU 推理）
    - 质量不如 OpenAI（但够用）

    常用模型：
    - BAAI/bge-large-zh: 中文效果好
    - sentence-transformers/all-MiniLM-L6-v2: 速度快

    类比 Java:
      // 加载本地模型
      SentenceTransformer model = new SentenceTransformer("BAAI/bge-large-zh");
      float[] vector = model.encode("你的文本");
    """
    print("\n>>> 使用本地 Embedding（HuggingFace）")

    try:
        # --- 【Python 语法】HuggingFaceEmbedding ---
        # 创建本地 Embedding 实例
        # model_name 指定要下载的模型
        local_embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

        # --- 【Python 语法】get_text_embedding() ---
        # 对文本进行编码
        text = "什么是向量数据库？"
        vector = local_embed.get_text_embedding(text)

        print(f"文本: {text}")
        print(f"向量维度: {len(vector)}")
        print(f"向量前 10 个值: {vector[:10]}")

    except ImportError:
        print("  [跳过] 需要安装 sentence-transformers")
        print("  运行: pip install sentence-transformers")


# ============================================================
# 4. 向量存储 — SimpleVectorStore
# ============================================================

def demo_simple_vector_store():
    """
    SimpleVectorStore 是 LlamaIndex 的内存向量存储

    特点：
    - 数据存储在内存中，重启后丢失
    - 适合学习和测试
    - 支持基本的向量搜索

    类比 Java:
      // 内存中的 Map<UUID, Vector>
      Map<String, float[]> vectorStore = new HashMap<>();
      vectorStore.put(uuid, vector);
      // 搜索最近邻
      List<Entry<String, float[]>> results = search(queryVector, k);
    """
    print("\n>>> 使用 SimpleVectorStore")

    from llama_index.core.vector_stores import SimpleVectorStore

    # --- 【Python 语法】SimpleVectorStore ---
    # 创建内存向量存储
    vector_store = SimpleVectorStore()

    # --- 【Python 语法】add() ---
    # 添加向量到存储
    # 参数：
    #   - doc_id: 文档 ID
    #   - embedding: 向量（list[float]）
    #   - metadata: 元数据（dict）
    # 类比 Java: vectorStore.add(docId, embedding, metadata);

    # 添加几条向量
    texts = [
        "猫是一种可爱的宠物",
        "狗是人类最好的朋友",
        "鱼生活在水中",
        "鸟在天空中飞翔",
    ]
    # 用随机向量模拟（实际应该用 Embedding 模型生成）
    np.random.seed(42)
    for i, text in enumerate(texts):
        embedding = np.random.rand(128).tolist()  # 128 维随机向量
        vector_store.add(
            doc_id=f"doc_{i}",
            embedding=embedding,
            metadata={"text": text},
        )
        print(f"  添加: {text}")

    # --- 【Python 语法】search() ---
    # 搜索最相似的向量
    # 参数：
    #   - query_embedding: 查询向量
    #   - similarity_top_k: 返回 Top-K 个结果
    # 类比 Java: vectorStore.search(queryVector, 3);
    query_embedding = np.random.rand(128).tolist()
    results = vector_store.search(query_embedding, similarity_top_k=3)

    print(f"\n搜索结果（Top-3）:")
    for i, result in enumerate(results):
        print(f"  #{i+1} 相似度: {result.similarity:.4f}")
        print(f"      文本: {result.text}")


# ============================================================
# 5. 向量存储 — ChromaDB
# ============================================================

def demo_chromadb():
    """
    ChromaDB 是轻量级向量数据库

    特点：
    - 持久化存储（数据不会丢失）
    - 支持多种索引算法（IVF, HNSW, FAISS）
    - 内置 API 服务
    - 适合中小规模数据（< 1000 万向量）

    类比 Java:
      // 类似嵌入式数据库（如 H2、SQLite）
      ChromaClient client = new ChromaClient();
      Collection collection = client.getOrCreateCollection("my_collection");
      collection.add(ids, embeddings, metadatas);
      Collection results = collection.query(query_embeddings, n_results=5);
    """
    print("\n>>> 使用 ChromaDB")

    try:
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        # --- 【Python 语法】chromadb.Client() ---
        # 创建 ChromaDB 客户端
        # persist_directory 指定持久化目录
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db"  # 数据持久化到 ./chroma_db 目录
        )

        # --- 【Python 语法】ChromaVectorStore ---
        # 创建 ChromaDB 向量存储包装器
        chroma_collection = chroma_client.get_or_create_collection("my_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # --- 【Python 语法】add() ---
        # 添加向量
        vector_store.add([
            "猫是一种可爱的宠物",
            "狗是人类最好的朋友",
            "鱼生活在水中",
        ])

        print("  ✓ ChromaDB 向量已添加")

        # --- 【Python 语法】query() ---
        # 搜索
        results = vector_store.query("动物", similarity_top_k=2)
        print(f"\n  搜索结果（Top-2）:")
        for r in results:
            print(f"    文本: {r.text}")

    except ImportError:
        print("  [跳过] 需要安装 chromadb")
        print("  运行: pip install chromadb")


# ============================================================
# 6. 向量存储 — FAISS
# ============================================================

def demo_faiss():
    """
    FAISS 是 Facebook 开发的向量检索库

    特点：
    - 极速搜索（GPU 加速）
    - 内存效率高
    - 支持多种索引类型（IVF, HNSW, PQ）
    - 适合大规模向量检索

    类比 Java:
      // 类似 Lucene 之于全文检索
      IndexFlatL2 index = new IndexFlatL2(dim);
      index.add(embeddings);
      index.search(query, k);
    """
    print("\n>>> 使用 FAISS")

    try:
        from llama_index.vector_stores.faiss import FaissVectorStore
        import faiss

        # --- 【Python 语法】faiss.IndexFlatL2 ---
        # 创建 FAISS 索引（L2 距离 = 欧氏距离）
        # 参数：向量维度
        dim = 1536  # OpenAI embedding 的维度
        faiss_index = faiss.IndexFlatL2(dim)

        # --- 【Python 语法】FaissVectorStore ---
        # 创建 FAISS 向量存储
        vector_store = FaissVectorStore(faiss_index)

        # 添加向量
        texts = [
            "向量数据库用于存储和检索高维向量",
            "Embedding 将文本转换为向量",
            "相似度搜索找到最相似的向量",
        ]
        # 用随机向量模拟
        np.random.seed(42)
        for text in texts:
            embedding = np.random.rand(dim).astype(np.float32).tolist()
            vector_store.add(text, embedding)

        print("  ✓ FAISS 向量已添加")

        # 搜索
        query_embedding = np.random.rand(dim).astype(np.float32).tolist()
        results = vector_store.query(query_embedding, similarity_top_k=2)
        print(f"\n  搜索结果（Top-2）:")
        for r in results:
            print(f"    文本: {r.text}")

    except ImportError:
        print("  [跳过] 需要安装 faiss-cpu")
        print("  运行: pip install faiss-cpu")


# ============================================================
# 7. 向量存储对比
# ============================================================

def compare_stores():
    """
    三种向量存储对比

    选择建议：
    - 学习/测试 → SimpleVectorStore（最简单）
    - 中小项目 → ChromaDB（持久化 + 易用）
    - 大规模 → FAISS（极速 + 内存优化）
    """
    print("\n" + "=" * 60)
    print("  向量存储对比")
    print("=" * 60)
    print("""
    ┌──────────────┬──────────────┬────────────┬──────────────┐
    │    存储       │   持久化     │   规模     │   适用场景    │
    ├──────────────┼──────────────┼────────────┼──────────────┤
    │ Simple       │ ❌ 内存      │ < 10万     │ 学习/测试     │
    │ ChromaDB     │ ✅ 磁盘      │ < 1000万   │ 中小项目      │
    │ FAISS        │ ✅ 磁盘      │ < 1亿      │ 大规模检索    │
    │ Milvus       │ ✅ 磁盘      │ > 1亿      │ 生产环境      │
    └──────────────┴──────────────┴────────────┴──────────────┘
    """)


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数：依次演示各种 Embedding 和向量存储
    """
    explain_embedding()
    demo_openai_embedding()
    demo_local_embedding()
    demo_simple_vector_store()
    demo_chromadb()
    demo_faiss()
    compare_stores()

    print("\n" + "=" * 60)
    print("  OK Embedding 与向量存储完成！")
    print("=" * 60)
    print("""
下一步：
  - 理解 Embedding 如何将文本转为向量
  - 选择合适的向量存储（ChromaDB 推荐用于学习）
  - 下一课学习如何构建索引
    """)


if __name__ == "__main__":
    main()
