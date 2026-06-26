"""
===============================================================================
 第 5 课：Index + Retriever — 构建索引与检索（RAG 的核心）
===============================================================================

【这一课学什么？】
  前两课我们学会了：
    - 第 3 课：用 Reader 把文件变成 Document
    - 第 1 课：用 NodeParser 把 Document 切成 Node
    - 第 4 课：把 Node 存入 Vector Store

  但这还不够——我们需要一个"管理层"来统一这些操作。
  Index 就是这个管理层：它把 Node + VectorStore 封装成一个可以查询的对象。
  Retriever 是 Index 的"搜索接口"，负责根据用户的问题找到最相关的 Node。

【类比 Java】
  Index ≈ JpaRepository + 搜索引擎
  - VectorStoreIndex.from_documents() ≈ 建索引 + 插入数据
  - index.as_retriever() ≈ 创建搜索服务
  - retriever.retrieve(query) ≈ repository.search(keyword)

【核心概念】
  VectorStoreIndex 的工作流程：
    Documents → (自动切分 + 向量化 + 存入 VectorStore) → Index
    Index → as_retriever() → Retriever
    Retriever.retrieve(query) → List[NodeWithScore]

  最关键的 API：
    index = VectorStoreIndex.from_documents(documents)  ← 一行搞定一切！
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("用户的问题")

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/05_Index与Retriever.py

【前置知识】
  - 第 1 课：Node / 文本切片
  - 第 2 课：Settings / 全局配置
  - 第 3 课：Reader / 文档加载
  - 第 4 课：Vector Store / 向量存储
"""

import os
import shutil
from pathlib import Path
from llama_index.core import Settings


# ============================================================================
# 第 1 节：VectorStoreIndex.from_documents() — 一行搞定
# ============================================================================

def demo_index_from_documents():
    """
    这是 LlamaIndex 最核心的 API

    VectorStoreIndex.from_documents() 会自动完成以下步骤：
    1. 如果没有配置 NodeParser，使用默认的 TokenTextSplitter
    2. 把每个 Document 切分成 Node
    3. 调用 Settings.embed_model 把每个 Node 的文本转成向量
    4. 把向量 + 文本 + 元数据存入 VectorStore
    5. 返回一个可用的 Index 对象

    类比 Java：
      // 手动方式（太麻烦）
      List<Node> nodes = splitter.split(documents);
      for (Node node : nodes) {
          float[] vec = embedModel.encode(node.getText());
          vectorStore.add(vec, node.getText(), node.getMetadata());
      }
      Index index = new VectorStoreIndex(nodes, vectorStore);

      // 一行搞定
      Index index = VectorStoreIndex.fromDocuments(documents);
    """
    print("=" * 60)
    print("【VectorStoreIndex.from_documents() 一行搞定】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 创建一些测试文档
    print("\n  --- 创建测试文档 ---")
    documents = [
        Document(
            text="公司实行每日八小时工作制。上午9点至12点，下午1点半至5点半。",
            metadata={"title": "考勤制度", "chapter": "工作时间"}
        ),
        Document(
            text="员工每年享有5-15天带薪年假。工作满1年5天，满10年10天，满20年15天。",
            metadata={"title": "年假制度", "chapter": "假期"}
        ),
        Document(
            text="迟到早退超过30分钟按旷工半天处理。每月有3次迟到豁免机会。",
            metadata={"title": "考勤制度", "chapter": "纪律"}
        ),
    ]
    print(f"    创建了 {len(documents)} 个 Document")

    # 核心：一行创建 Index
    # 注意：这里没有传 vector_store！
    # 因为 from_documents() 会自动创建一个内存中的 SimpleVectorStore
    print("\n  --- 创建 VectorStoreIndex ---")
    index = VectorStoreIndex.from_documents(documents)
    print(f"    ✓ Index 创建完成")
    print(f"    包含文档数: {len(index.docstore.docs)}")

    # 查看 Index 内部结构
    print("\n  --- Index 内部结构 ---")
    print(f"    docstore 中的文档数: {len(list(index.docstore.docs.values()))}")
    print(f"    使用的向量存储: {type(index.vector_store).__name__}")

    # 测试查询
    print("\n  --- 测试查询 ---")
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("年假几天？")
    print(f"    查询: '年假几天？'")
    print(f"    找到 {len(nodes)} 个相关节点:")
    for i, node_score in enumerate(nodes):
        print(f"      结果 {i + 1}: {node_score.node.text[:40]}... (分数: {node_score.score:.4f})")


# ============================================================================
# 第 2 节：Retriever — 检索器详解
# ============================================================================

def demo_retriever():
    """
    Retriever 是 Index 的搜索接口

    它的工作：
    1. 接收用户查询（QueryBundle）
    2. 把查询文本转成向量（使用 Settings.embed_model）
    3. 在 VectorStore 中做相似度搜索
    4. 返回最相关的 Node 列表（带分数）

    类比 Java：
      public List<NodeWithScore> retrieve(String query) {
          float[] queryVec = embedModel.encode(query);
          return vectorStore.search(queryVec, topK);
      }
    """
    print("=" * 60)
    print("【Retriever 详解】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 准备文档
    documents = [
        Document(text="公司实行每日八小时工作制。", metadata={"dept": "HR"}),
        Document(text="员工年薪范围15-50万。", metadata={"dept": "Finance"}),
        Document(text="每周周五下午团队团建活动。", metadata={"dept": "Team"}),
        Document(text="入职缴纳五险一金。", metadata={"dept": "HR"}),
        Document(text="年终奖2-6个月工资。", metadata={"dept": "Finance"}),
    ]

    # 创建 Index
    index = VectorStoreIndex.from_documents(documents)

    # 方式 1：使用 as_retriever() 创建检索器
    print("\n  --- 方式 1: as_retriever() ---")
    retriever = index.as_retriever(similarity_top_k=3)
    # similarity_top_k = 返回最相似的前 k 个结果
    # 类比 Java: query.setMaxResults(3);

    nodes = retriever.retrieve("福利待遇怎么样？")
    print(f"    查询: '福利待遇怎么样？'")
    print(f"    返回 {len(nodes)} 个结果:\n")
    for i, node_score in enumerate(nodes):
        node = node_score.node
        score = node_score.score
        print(f"    结果 {i + 1}:")
        print(f"      文本: {node.text}")
        print(f"      相似度: {score:.4f}")
        print(f"      部门: {node.metadata.get('dept')}")
        print()

    # 方式 2：自定义检索器（调整参数）
    print("  --- 方式 2: 调整检索参数 ---")
    retriever_custom = index.as_retriever(
        similarity_top_k=2,       # 只返回前 2 个
        similarity_cutoff=0.5,    # 只返回相似度 > 0.5 的结果
    )
    nodes_custom = retriever_custom.retrieve("工作时间是多久？")
    print(f"    查询: '工作时间是多久？' (top_k=2, cutoff=0.5)")
    print(f"    返回 {len(nodes_custom)} 个结果:")
    for node_score in nodes_custom:
        print(f"      - {node_score.node.text} (分数: {node_score.score:.4f})")

    # 方式 3：使用不同的检索模式
    print("\n  --- 方式 3: 不同的检索模式 ---")
    print("""
    检索模式（mode 参数）：
      - "default"    → 向量相似度搜索（最常用）
      - "sparse"     → 关键词匹配（TF-IDF）
      - "hybrid"     → 向量 + 关键词混合（最准确）

    类比 Java：
      // default: SELECT * FROM nodes ORDER BY cosine_similarity(vec, ?) LIMIT 3
      // sparse:  SELECT * FROM nodes WHERE text LIKE '%工作%'
      // hybrid:  SELECT * FROM nodes ORDER BY (vec_score * 0.6 + keyword_score * 0.4) LIMIT 3
    """)


# ============================================================================
# 第 3 节：NodeWithScore — 检索结果的包装
# ============================================================================

def demo_node_with_score():
    """
    retriever.retrieve() 返回的是 List[NodeWithScore]

    NodeWithScore 是一个包装类，包含：
    - node      → 实际的 Node 对象（文本 + 元数据）
    - score     → 相似度分数（0-1 之间，越高越相关）
    - metadata  → 额外的检索元数据

    类比 Java：
      public class NodeWithScore {
          private Node node;
          private Double score;
          // getters...
      }
    """
    print("=" * 60)
    print("【NodeWithScore 详解】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    documents = [
        Document(text="猫是哺乳动物，喜欢抓老鼠。"),
        Document(text="狗是人类最好的朋友。"),
        Document(text="Python 是一门编程语言。"),
    ]

    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=2)
    results = retriever.retrieve("动物")

    print("\n  retriever.retrieve() 返回的类型: List[NodeWithScore]")
    print(f"  返回了 {len(results)} 个 NodeWithScore 对象:\n")

    for i, nws in enumerate(results):
        print(f"  NodeWithScore {i + 1}:")
        print(f"    类型: {type(nws).__name__}")
        print(f"    .node.text      = '{nws.node.text}'")
        print(f"    .score          = {nws.score:.4f}")
        print(f"    .node.id_       = {nws.node.id_}")
        print(f"    .node.metadata  = {dict(nws.node.metadata)}")
        print()


# ============================================================================
# 第 4 节：Index 的持久化与加载
# ============================================================================

def demo_index_persistence():
    """
    索引可以保存到磁盘，下次直接加载，不用重新构建

    类比 Java：
      // 保存
      entityManager.persist(index);
      transaction.commit();

      // 加载
      Index loaded = entityManager.find(Index.class, indexId);
    """
    print("=" * 60)
    print("【Index 的持久化与加载】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.storage import StorageContext

    documents = [
        Document(text="公司每天工作8小时。"),
        Document(text="年假5-15天。"),
    ]

    persist_dir = "./index_persist"

    # 第 1 步：创建并保存 Index
    print("\n  --- 第 1 步：创建并保存 Index ---")
    index = VectorStoreIndex.from_documents(documents)
    print(f"    ✓ 创建了 Index，包含 {len(index.docstore.docs)} 个文档")

    # 持久化
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"    ✓ Index 已保存到: {persist_dir}")

    # 第 2 步：从磁盘加载 Index
    print("\n  --- 第 2 步：从磁盘加载 Index ---")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    loaded_index = VectorStoreIndex.from_vector_space(storage_context)
    print(f"    ✓ Index 已从 {persist_dir} 加载")
    print(f"    包含文档数: {len(loaded_index.docstore.docs)}")

    # 第 3 步：加载后的 Index 可以正常使用
    print("\n  --- 第 3 步：加载后查询 ---")
    retriever = loaded_index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("工作时间")
    for node_score in nodes:
        print(f"    结果: {node_score.node.text} (分数: {node_score.score:.4f})")

    # 清理
    shutil.rmtree(persist_dir, ignore_errors=True)
    print(f"\n    ✓ 已清理持久化目录")


# ============================================================================
# 第 5 节：Index 的不同类型
# ============================================================================

def demo_index_types():
    """
    LlamaIndex 提供了多种 Index 类型，每种适合不同的场景

    最常见的是 VectorStoreIndex，但也有其他选择。
    """
    print("=" * 60)
    print("【Index 类型对比】")
    print("=" * 60)

    print("""
  ┌────────────────────────┬────────────┬──────────────┬────────────────────┐
  │ Index 类型             │ 检索方式   │ 适用场景      │ Java 类比           │
  ├────────────────────────┼────────────┼──────────────┼────────────────────┤
  │ VectorStoreIndex       │ 向量相似度  │ 语义搜索(最常用) │ 全文搜索引擎       │
  │ SummaryIndex           │ 遍历所有    │ 短文档直接回答  │ List.get(i)        │
  │ TreeIndex              │ 树形遍历    │ 层次化文档     │ 文件系统目录树      │
  │ KeywordTableIndex      │ 关键词匹配  │ 实体识别       │ HashMap.get(key)   │
  │ KnowledgeGraphIndex    │ 图遍历      │ 关系推理       │ Neo4j              │
  │ PropertyGraphIndex     │ 属性图查询  │ 结构化关系     │ PropertyGraph DB   │
  └────────────────────────┴────────────┴──────────────┴────────────────────┘

  最常用的是 VectorStoreIndex（90% 的场景都用它）。
  其他 Index 在后续课程中会详细介绍。
  """)


# ============================================================================
# 第 6 节：完整示例 — 从文件到检索的全流程
# ============================================================================

def demo_full_workflow():
    """
    完整演示从文件到检索的每一步

    这是 RAG 的核心管线，理解了它就理解了 LlamaIndex。
    """
    print("=" * 60)
    print("【完整工作流：文件 → 索引 → 检索】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 第 1 步：模拟读取文件（实际项目中用 Reader）
    print("\n  --- 第 1 步：加载文档 ---")
    documents = [
        Document(
            text="公司实行每日八小时工作制，上午9:00-12:00，下午13:30-17:30。",
            metadata={"file": "考勤制度.pdf", "page": 1}
        ),
        Document(
            text="员工每年享有5天带薪年假，工作满10年增至10天，满20年增至15天。",
            metadata={"file": "年假制度.pdf", "page": 1}
        ),
        Document(
            text="病假需提供二级以上医院诊断证明，病假期间按基本工资80%发放。",
            metadata={"file": "病假制度.pdf", "page": 1}
        ),
        Document(
            text="员工入职即缴纳五险一金，公积金比例个人12%公司12%。",
            metadata={"file": "福利制度.pdf", "page": 1}
        ),
        Document(
            text="年终奖根据个人绩效评定：A级6个月，B级3个月，C级1个月。",
            metadata={"file": "薪酬制度.pdf", "page": 1}
        ),
    ]
    print(f"    加载了 {len(documents)} 个 Document")

    # 第 2 步：创建 Index（自动切分 + 向量化 + 存储）
    print("\n  --- 第 2 步：创建 VectorStoreIndex ---")
    index = VectorStoreIndex.from_documents(documents)
    print(f"    ✓ Index 构建完成")
    print(f"    文档数: {len(list(index.docstore.docs.values()))}")

    # 第 3 步：创建检索器
    print("\n  --- 第 3 步：创建 Retriever ---")
    retriever = index.as_retriever(similarity_top_k=2)
    print(f"    ✓ Retriever 创建完成 (top_k=2)")

    # 第 4 步：执行查询
    print("\n  --- 第 4 步：执行查询 ---")
    queries = [
        "年假几天？",
        "迟到怎么处罚？",
        "五险一金怎么交？",
    ]

    for query in queries:
        print(f"\n    查询: '{query}'")
        nodes = retriever.retrieve(query)
        for i, node_score in enumerate(nodes):
            print(f"      结果 {i + 1}: {node_score.node.text[:50]}...")
            print(f"        分数: {node_score.score:.4f}")
            print(f"        来源: {node_score.node.metadata.get('file')}")


# ============================================================================
# 第 7 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 5 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. VectorStoreIndex.from_documents() 一行搞定 Document → Index          │
│  2. Index 自动完成：切分 → 向量化 → 存入 VectorStore                    │
│  3. index.as_retriever() 创建搜索接口                                    │
│  4. retriever.retrieve(query) 返回 NodeWithScore 列表                    │
│  5. Index 可以持久化和加载                                              │
│  6. 工作流：Documents → Index → Retriever → Nodes                       │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 创建 Index（一行搞定）                                                │
│  from llama_index.core.indices.vector_store import VectorStoreIndex     │
│  index = VectorStoreIndex.from_documents(documents)                      │
│                                                                         │
│  # 创建检索器                                                           │
│  retriever = index.as_retriever(similarity_top_k=3)                     │
│                                                                         │
│  # 执行查询                                                             │
│  results = retriever.retrieve("用户的问题")                              │
│  for node_score in results:                                             │
│      print(node_score.node.text)                                        │
│      print(node_score.score)                                            │
│                                                                         │
│  # 持久化                                                               │
│  index.storage_context.persist(persist_dir="./my_index")                │
│  loaded = VectorStoreIndex.from_vector_space(                            │
│      StorageContext.from_defaults(persist_dir="./my_index"))            │
│                                                                         │
│  【下一课预告】                                                           │
│  第 6 课：Postprocessor / 重排序 — 提升检索质量                          │
│  类比 Java：二级排序 / 相关性打分                                       │
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
║   第 5 课：Index + Retriever — 构建索引与检索             ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. VectorStoreIndex.from_documents() 一行搞定          ║
║   2. Retriever 详解                                      ║
║   3. NodeWithScore 详解                                  ║
║   4. Index 的持久化与加载                                ║
║   5. Index 类型对比                                      ║
║   6. 完整工作流演示                                      ║
║   7. 总结                                               ║
║                                                          ║
║   前置知识：第 1-4 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：from_documents() 一行搞定 ━━━")
    demo_index_from_documents()

    print("\n━━━ 第 2 节：Retriever 详解 ━━━")
    demo_retriever()

    print("\n━━━ 第 3 节：NodeWithScore 详解 ━━━")
    demo_node_with_score()

    print("\n━━━ 第 4 节：Index 持久化 ━━━")
    demo_index_persistence()

    print("\n━━━ 第 5 节：Index 类型对比 ━━━")
    demo_index_types()

    print("\n━━━ 第 6 节：完整工作流 ━━━")
    demo_full_workflow()

    print("\n🎉 第 5 课完成！")
    print("   建议下一步：阅读 week03/code/p27-index-retriever.ipynb")
