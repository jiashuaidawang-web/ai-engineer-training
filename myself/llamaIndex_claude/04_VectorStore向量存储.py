"""
===============================================================================
 第 4 课：Vector Store / 向量存储 — FAISS 持久化向量
===============================================================================

【这一课学什么？】
  第 3 课我们用 Reader 把文件变成了 Document，第 1 课把 Document 切成了 Node。
  但这些 Node 还只是内存中的对象。要把它们存起来供后续检索，就需要 Vector Store。
  Vector Store 是存储 Node 的向量嵌入（embedding）的地方，支持相似度搜索。

【类比 Java】
  Vector Store ≈ 数据库（MySQL / MongoDB / Elasticsearch）
  - 传统数据库：存储结构化数据，支持 SQL 查询
  - Vector Store：存储向量数据，支持相似度查询（"找最相似的 N 个"）
  StorageContext ≈ DataSource 配置（告诉程序用哪个数据库）

【核心概念】
  Vector Store 的工作流程：
    Node → Embedding（向量化） → 存入 Vector Store → 相似度搜索

  常用 Vector Store：
    - FAISS        → Meta 开源，速度快，适合本地开发（本课重点）
    - Chroma       → 零配置，嵌入式，适合原型开发
    - Milvus       → 分布式，生产级，支持海量数据
    - Elasticsearch → 混合搜索（关键词 + 向量）
    - Pinecone     → 云服务，免运维

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/04_VectorStore向量存储.py

【前置知识】
  - 第 1 课：Node / 文本切片
  - 第 2 课：Settings / 全局配置
  - 第 3 课：Reader / 文档加载
"""

import os
import shutil
from pathlib import Path
from llama_index.core import Settings


# ============================================================================
# 第 1 节：FAISS — 快速入门
# ============================================================================

def demo_faiss_basic():
    """
    FAISS (Facebook AI Similarity Search) 是最常用的本地向量存储

    特点：
    - 速度极快（C++ 底层优化）
    - 内存占用小
    - 支持 L2 距离和余弦相似度
    - 数据持久化到磁盘

    类比 Java：
      // 伪代码
      FaissVectorStore store = new FaissVectorStore(dimensions=1536);
      store.add(embedding, text, metadata);
      List<Result> results = store.similaritySearch(query_embedding, topK=5);
    """
    print("=" * 60)
    print("【FAISS 基础演示】")
    print("=" * 60)

    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import TextNode, Document
    from llama_index.core.node_parser import TokenTextSplitter

    # 第 1 步：创建 FAISS 向量存储
    # SimpleVectorStore 是 LlamaIndex 提供的轻量级内存存储
    # 如果需要真正的 FAISS，需要安装：pip install llama-index-vector-stores-faiss
    print("\n  --- 创建向量存储 ---")
    vector_store = SimpleVectorStore()
    print("    ✓ 创建了 SimpleVectorStore（内存模式）")

    # 第 2 步：准备一些 Node（用第 1 课的 NodeParser 知识）
    print("\n  --- 准备 Node 数据 ---")
    doc = Document(
        text="""
        公司实行每日八小时工作制。
        上午工作时间为九时至十二时，中午休息一小时。
        下午工作时间为十三时三十分至十七时三十分。
        员工每周享有两天休息日，通常为周六和周日。
        因工作需要安排加班的，应优先安排补休。
        无法安排补休的，按国家规定支付加班工资。
        """,
        metadata={"title": "考勤制度"}
    )

    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"    ✓ 创建了 {len(nodes)} 个 Node")

    # 第 3 步：将 Node 加入向量存储
    # add() 方法接收 Node 列表，内部会自动：
    #   1. 调用 Settings.embed_model 将文本转为向量
    #   2. 将向量 + 文本 + 元数据一起存入 store
    print("\n  --- 将 Node 存入向量存储 ---")
    vector_store.add(nodes)
    print("    ✓ 已将 Node 存入 vector_store")

    # 第 4 步：查询向量存储（相似度搜索）
    # 这是 Vector Store 的核心功能：给定一个查询文本，
    # 找到存储中与它最相似的那些 Node
    print("\n  --- 查询向量存储 ---")
    query = "员工每天工作几个小时？"

    # 方式 1：使用 SimpleVectorStore 的 query 方法
    # 参数：
    #   query_text  → 查询文本
    #   similarity_top_k → 返回最相似的前 k 个结果
    results = vector_store.query(query, similarity_top_k=3)
    print(f"    查询: '{query}'")
    print(f"    找到 {len(results)} 个相似节点:\n")

    for i, result in enumerate(results):
        node = result["node"]
        score = result["similarity"]  # 相似度分数（越高越相关）
        print(f"    结果 {i + 1}:")
        print(f"      文本: {node.text.strip()[:50]}...")
        print(f"      相似度: {score:.4f}")
        print(f"      元数据: {dict(node.metadata)}")
        print()


# ============================================================================
# 第 2 节：StorageContext — 存储上下文
# ============================================================================

def demo_storage_context():
    """
    StorageContext 是一个容器，管理所有的存储后端

    它持有：
    - docstore      → 存储原始 Document
    - index_store   → 存储索引元数据
    - vector_store  → 存储向量嵌入
    - graph_store   → 存储知识图谱（可选）

    类比 Java：
      // Spring 的 ApplicationContext 管理所有 Bean
      @Component
      public class StorageContext {
          private DocStore docStore;
          private IndexStore indexStore;
          private VectorStore vectorStore;
          private GraphStore graphStore;
      }
    """
    print("=" * 60)
    print("【StorageContext — 存储上下文】")
    print("=" * 60)

    from llama_index.core.storage import StorageContext
    from llama_index.core.vector_stores import SimpleVectorStore

    print("\n  --- 创建 StorageContext ---")

    # 方式 1：从零创建（开发时常用）
    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("    ✓ 方式 1: 从简单的 vector_store 创建")

    # 方式 2：从持久化目录加载（生产环境常用）
    # 类比 Java: 从配置文件加载 DataSource
    persist_dir = "./faiss_persist"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # 先保存
    storage_context.persist(persist_dir=persist_dir)
    print(f"    ✓ StorageContext 已持久化到: {persist_dir}")

    # 再加载
    loaded_context = StorageContext.from_defaults(persist_dir=persist_dir)
    print(f"    ✓ StorageContext 已从 {persist_dir} 加载")

    # 清理
    shutil.rmtree(persist_dir)
    print(f"    ✓ 已清理持久化目录")

    print("""
  💡 StorageContext 的作用：
    - 统一管理所有存储后端（向量库、文档库、图数据库等）
    - 支持持久化到磁盘（类比 Java 的对象序列化）
    - 支持从磁盘恢复（类比 Java 的反序列化）
    """)


# ============================================================================
# 第 3 节：向量存储 vs 传统数据库
# ============================================================================

def demo_vector_vs_traditional():
    """
    用对比的方式理解向量存储与传统数据库的区别

    这是很多 Java 程序员最难跨越的思维障碍 —— 从"精确匹配"到"相似度匹配"。
    """
    print("=" * 60)
    print("【向量存储 vs 传统数据库】")
    print("=" * 60)

    print("""
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │ 特性               │ 传统数据库 (MySQL)   │ 向量数据库 (FAISS)   │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ 查询方式            │ WHERE name = '张三'  │ 找最相似的 N 个向量   │
  │ 匹配类型            │ 精确匹配             │ 近似匹配（相似度）    │
  │ 存储内容            │ 结构化数据           │ 文本 + 向量 + 元数据  │
  │ 索引类型            │ B-Tree / Hash       │ IVF / HNSW / PQ      │
  │ 速度                │ 快（有索引时）       │ 极快（FAISS 用 C++） │
  │ 适用场景            │ 订单、用户、交易     │ 语义搜索、推荐系统    │
  │ Java 类比           │ JPA Repository      │ 无直接等价           │
  └─────────────────────┴──────────────────────┴──────────────────────┘

  示例对比：

  MySQL 查询：
    SELECT * FROM articles WHERE title LIKE '%考勤%';
    → 只能找到标题中包含"考勤"的文章

  FAISS 查询：
    输入: "员工每天工作多久？"
    → 能找到包含"八小时工作制"、"上午九点到十二点"的文章
    → 即使这些文章里没有"考勤"两个字！
    → 这就是语义搜索的力量。
    """)


# ============================================================================
# 第 4 节：向量嵌入（Embedding）详解
# ============================================================================

def demo_embedding_details():
    """
    Embedding 是向量存储的核心 —— 没有 embedding，就没有向量存储

    这一节深入理解 embedding 是怎么工作的。
    """
    print("=" * 60)
    print("【Embedding 详解】")
    print("=" * 60)

    # 检查是否有 LLM 配置
    if Settings.embed_model is None:
        print("  ⚠️  Settings.embed_model 未配置，先配置一个...\n")
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            from llama_index.embeddings.dashscope import DashScopeEmbedding
            from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels
            Settings.embed_model = DashScopeEmbedding(
                model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
                api_key=api_key,
            )
            print(f"  ✓ 已配置 Embedding: {Settings.embed_model.model_name}\n")
        else:
            print("  ⚠️  缺少 DASHSCOPE_API_KEY，跳过 embedding 演示\n")
            return

    embed_model = Settings.embed_model

    # 测试 1：把文本变成向量
    print("  --- 测试 1：文本 → 向量 ---")
    texts = [
        "猫喜欢吃鱼",
        "狗狗是人类最好的朋友",
        "Python 是一门编程语言",
    ]

    for text in texts:
        # get_text_embedding() 返回一个浮点数列表
        # 类比 Java: float[] vec = embedModel.encode(text);
        vec = embed_model.get_text_embedding(text)
        print(f"    '{text}' → {len(vec)} 维向量")
        print(f"      前 5 个值: {[round(v, 4) for v in vec[:5]]}")

    # 测试 2：批量获取 embedding（更高效）
    print("\n  --- 测试 2：批量获取 embedding ---")
    # get_text_embeddings() 一次处理多条文本
    vectors = embed_model.get_text_embeddings(texts)
    print(f"    批量处理 {len(texts)} 条文本，得到 {len(vectors)} 个向量")
    print(f"    每个向量的维度: {len(vectors[0])}")

    # 测试 3：理解向量距离 = 语义相似度
    print("\n  --- 测试 3：向量距离 = 语义相似度 ---")
    # 语义相近的文本，它们的向量距离也更近
    # 类比 Java: double dist = cosineDistance(vec1, vec2);

    vec_cat = embed_model.get_text_embedding("猫喜欢吃鱼")
    vec_dog = embed_model.get_text_embedding("狗狗喜欢吃肉")
    vec_code = embed_model.get_text_embedding("Java 是一门编程语言")

    # 计算余弦相似度（简化版，用向量差的平方和近似）
    def cosine_similarity(a, b):
        """计算两个向量的余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    sim_cat_dog = cosine_similarity(vec_cat, vec_dog)
    sim_cat_code = cosine_similarity(vec_cat, vec_code)

    print(f"    '猫喜欢吃鱼' vs '狗狗喜欢吃肉': 相似度 = {sim_cat_dog:.4f}")
    print(f"    '猫喜欢吃鱼' vs 'Java 是一门编程语言': 相似度 = {sim_cat_code:.4f}")
    print(f"\n    可以看到：猫的文本和狗的文本更相似！")
    print(f"    这就是向量存储做语义搜索的基础。")


# ============================================================================
# 第 5 节：持久化 — 保存和加载向量存储
# ============================================================================

def demo_persistence():
    """
    演示如何将向量存储保存到磁盘，下次直接加载

    类比 Java：
      // 序列化
      ObjectOutputStream out = new ObjectOutputStream(
          new FileOutputStream("vector_store.dat"));
      out.writeObject(vectorStore);
      out.close();

      // 反序列化
      ObjectInputStream in = new ObjectInputStream(
          new FileInputStream("vector_store.dat"));
      VectorStore loaded = (VectorStore) in.readObject();
      in.close();
    """
    print("=" * 60)
    print("【向量存储的持久化】")
    print("=" * 60)

    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import TextNode
    import faiss

    # 创建 FAISS 向量存储
    # 注意：需要先安装 faiss-cpu: pip install faiss-cpu
    try:
        import faiss
        from llama_index.vector_stores.faiss import FaissVectorStore

        print("\n  --- 创建 FAISS 向量存储 ---")
        # faiss.IndexFlatL2 表示使用 L2 距离（欧几里得距离）
        # 1536 是 embedding 的维度（根据你用的模型调整）
        # 类比 Java: new FaissIndex(new L2Distance(), 1536)
        faiss_index = faiss.IndexFlatL2(1536)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        print("    ✓ 创建了 FAISS 向量存储")

        # 添加一些 Node
        print("\n  --- 添加 Node ---")
        nodes = [
            TextNode(text="猫是哺乳动物", metadata={"animal": "cat"}),
            TextNode(text="狗也是哺乳动物", metadata={"animal": "dog"}),
            TextNode(text="Python 是编程语言", metadata={"animal": "none"}),
        ]
        vector_store.add(nodes)
        print(f"    ✓ 添加了 {len(nodes)} 个 Node")

        # 持久化到磁盘
        print("\n  --- 持久化到磁盘 ---")
        persist_dir = "./faiss_db"
        vector_store.persist(persist_dir=persist_dir)
        print(f"    ✓ 已保存到: {persist_dir}")

        # 从磁盘加载
        print("\n  --- 从磁盘加载 ---")
        loaded_store = FaissVectorStore.from_persist_dir(persist_dir)
        print(f"    ✓ 已从 {persist_dir} 加载")

        # 查询
        print("\n  --- 加载后查询 ---")
        results = loaded_store.query("哺乳动物有哪些？", similarity_top_k=2)
        print(f"    查询: '哺乳动物有哪些？'")
        print(f"    找到 {len(results)} 个结果:")
        for r in results:
            node = r["node"]
            print(f"      - {node.text} (元数据: {dict(node.metadata)})")

        # 清理
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"\n    ✓ 已清理持久化目录")

    except ImportError:
        print("  ⚠️  faiss 未安装，使用 SimpleVectorStore 演示\n")

        # 使用 SimpleVectorStore 做持久化演示
        from llama_index.core.storage import StorageContext
        from llama_index.core.schema import TextNode

        persist_dir = "./simple_persist"

        # 创建并保存
        print("  --- 创建并保存 SimpleVectorStore ---")
        vector_store = SimpleVectorStore()
        nodes = [
            TextNode(text="猫是哺乳动物", metadata={"animal": "cat"}),
            TextNode(text="狗也是哺乳动物", metadata={"animal": "dog"}),
        ]
        vector_store.add(nodes)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.persist(persist_dir=persist_dir)
        print(f"    ✓ 已保存到: {persist_dir}")

        # 加载
        print("\n  --- 从磁盘加载 ---")
        loaded_context = StorageContext.from_defaults(persist_dir=persist_dir)
        print(f"    ✓ 已从 {persist_dir} 加载")

        # 清理
        shutil.rmtree(persist_dir, ignore_errors=True)
        print(f"    ✓ 已清理")


# ============================================================================
# 第 6 节：常用 Vector Store 对比
# ============================================================================

def demo_vector_store_comparison():
    """
    各种 Vector Store 的对比，帮你选择适合的
    """
    print("=" * 60)
    print("【Vector Store 对比表】")
    print("=" * 60)

    print("""
  ┌──────────────┬──────────┬──────────┬───────────┬──────────────┐
  │ Vector Store │ 速度     │ 容量     │ 持久化    │ 适用场景      │
  ├──────────────┼──────────┼──────────┼───────────┼──────────────┤
  │ FAISS        │ ★★★★★  │ 百万级   │ ✓         │ 本地开发/中小规模 │
  │ Chroma       │ ★★★★   │ 十万级   │ ✓         │ 快速原型/个人项目  │
  │ Milvus       │ ★★★★   │ 十亿级   │ ✓         │ 生产环境/大规模    │
  │ Elasticsearch│ ★★★    │ 百万级   │ ✓         │ 混合搜索（关键词+向量）│
  │ Pinecone     │ ★★★★   │ 百万级   │ 云托管    │ 不想运维的团队     │
  │ SQLite-VSS   │ ★★     │ 万级     │ ✓         │ 超小规模/测试      │
  │ SimpleVector │ ★★★    │ 千级     │ 部分      │ 学习/演示          │
  └──────────────┴──────────┴──────────┴───────────┴──────────────┘

  安装方式（类比 Maven 依赖）：
    pip install llama-index-vector-stores-faiss    # FAISS
    pip install llama-index-vector-stores-chroma   # Chroma
    pip install llama-index-vector-stores-milvus   # Milvus
    pip install llama-index-vector-stores-elasticsearch  # ES
    pip install pinecone                           # Pinecone (Python SDK)

  推荐学习路线：
    第 1 步：SimpleVectorStore（内置，无需安装）→ 理解概念
    第 2 步：FAISS → 本地开发首选，速度快
    第 3 步：Milvus/Elasticsearch → 生产环境
    """)


# ============================================================================
# 第 7 节：完整示例 — Reader → Node → VectorStore 全流程
# ============================================================================

def demo_full_flow():
    """
    把前三课的知识串起来：
    Reader 读取文件 → NodeParser 切分 → VectorStore 存储

    类比 Java：
      // 1. 读取文件
      String content = Files.readString(path);
      Document doc = new Document(content);

      // 2. 切分
      List<Node> nodes = splitter.split(doc);

      // 3. 存入向量数据库
      vectorStore.add(nodes);
    """
    print("=" * 60)
    print("【全流程示例：Reader → Node → VectorStore】")
    print("=" * 60)

    from llama_index.core.readers.simple_file_based import SimpleDirectoryReader
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import Document

    # 第 1 步：创建测试文件
    test_dir = Path("./test_full_flow")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "制度.txt").write_text(
        "公司实行每日八小时工作制。\n"
        "上午工作时间为九时至十二时。\n"
        "下午工作时间为十三时三十分至十七时三十分。\n"
        "员工每周享有两天休息日。\n"
        "年假：工作满1年5天，满10年10天，满20年15天。\n",
        encoding="utf-8"
    )

    print("\n  --- 第 1 步：Reader 读取文件 ---")
    reader = SimpleDirectoryReader(input_dir=str(test_dir), required_exts=[".txt"])
    documents = reader.load_data()
    print(f"    读取了 {len(documents)} 个 Document")

    print("\n  --- 第 2 步：NodeParser 切分 ---")
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"    切分成 {len(nodes)} 个 Node")

    print("\n  --- 第 3 步：存入 VectorStore ---")
    vector_store = SimpleVectorStore()
    vector_store.add(nodes)
    print(f"    ✓ Node 已存入向量存储")

    print("\n  --- 第 4 步：查询 ---")
    results = vector_store.query("年假几天？", similarity_top_k=2)
    print(f"    查询: '年假几天？'")
    for i, r in enumerate(results):
        node = r["node"]
        print(f"    结果 {i + 1}: {node.text.strip()}")

    # 清理
    shutil.rmtree(test_dir)
    print(f"\n  🎉 全流程演示完成！")


# ============================================================================
# 第 8 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 4 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. Vector Store 是存储 Node 向量的地方，支持相似度搜索                   │
│  2. FAISS 是最常用的本地向量存储，速度快                                  │
│  3. StorageContext 统一管理所有存储后端                                  │
│  4. Embedding 把文本变成向量，向量距离 = 语义相似度                       │
│  5. 可以持久化到磁盘，下次直接加载                                       │
│  6. 工作流：文件 → Reader → Document → NodeParser → Node → VectorStore  │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 创建 FAISS 向量存储                                                   │
│  import faiss                                                           │
│  from llama_index.vector_stores.faiss import FaissVectorStore           │
│  faiss_index = faiss.IndexFlatL2(1536)  # 1536 = embedding 维度        │
│  vector_store = FaissVectorStore(faiss_index=faiss_index)               │
│                                                                         │
│  # 添加 Node                                                            │
│  vector_store.add(nodes)                                                │
│                                                                         │
│  # 查询                                                                 │
│  results = vector_store.query("查询文本", similarity_top_k=5)           │
│                                                                         │
│  # 持久化                                                               │
│  vector_store.persist(persist_dir="./my_db")                            │
│  loaded = FaissVectorStore.from_persist_dir("./my_db")                  │
│                                                                         │
│  【下一课预告】                                                           │
│  第 5 课：Index + Retriever — 构建索引与检索                             │
│  类比 Java：JpaRepository + 搜索服务                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   第 4 课：Vector Store / 向量存储 — FAISS 持久化向量     ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. FAISS 基础演示                                      ║
║   2. StorageContext — 存储上下文                         ║
║   3. 向量存储 vs 传统数据库                              ║
║   4. Embedding 详解                                      ║
║   5. 持久化 — 保存和加载向量存储                         ║
║   6. 常用 Vector Store 对比                             ║
║   7. 完整示例：Reader → Node → VectorStore               ║
║   8. 总结                                               ║
║                                                          ║
║   前置知识：第 1-3 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：FAISS 基础 ━━━")
    demo_faiss_basic()

    print("\n━━━ 第 2 节：StorageContext ━━━")
    demo_storage_context()

    print("\n━━━ 第 3 节：向量存储 vs 传统数据库 ━━━")
    demo_vector_vs_traditional()

    print("\n━━━ 第 4 节：Embedding 详解 ━━━")
    demo_embedding_details()

    print("\n━━━ 第 5 节：持久化演示 ━━━")
    demo_persistence()

    print("\n━━━ 第 6 节：Vector Store 对比表 ━━━")
    demo_vector_store_comparison()

    print("\n━━━ 第 7 节：全流程示例 ━━━")
    demo_full_flow()

    print("\n🎉 第 4 课完成！")
    print("   建议下一步：阅读 week03/code/p26-vector-store.ipynb")
