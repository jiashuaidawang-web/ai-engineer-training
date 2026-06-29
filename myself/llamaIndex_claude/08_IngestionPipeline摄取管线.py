"""
===============================================================================
 第 8 课：Ingestion Pipeline / 摄取管线 — 一键完成整个数据流水线
===============================================================================

【这一课学什么？】
  到目前为止，我们学了：
    - 第 3 课：Reader 读取文件 → Document
    - 第 1 课：NodeParser 切分 Document → Node
    - 第 4 课：VectorStore 存储 Node 的向量
    - 第 5 课：Index 封装这些操作

  但每一步都要手动调用，代码很长。IngestionPipeline 把这些步骤
  串联成一个声明式的流水线，一行代码完成整个 ETL 过程。

【类比 Java】
  手动方式（第 1-7 课学的）：
    // 步骤 1：读取
    List<Document> docs = reader.load();
    // 步骤 2：切分
    List<Node> nodes = splitter.split(docs);
    // 步骤 3：向量化
    for (Node node : nodes) {
        float[] vec = embedModel.encode(node.getText());
        vectorStore.add(vec, node);
    }
    // 步骤 4：建索引
    Index index = new VectorStoreIndex(nodes, vectorStore);

  IngestionPipeline（一行搞定）：
    Pipeline pipeline = new Pipeline(reader, splitter, embedder, vectorStore);
    pipeline.run(documents);

  类比 Spring Batch：
    @BatchJob
    public void ingest() {
        stepReader()    // Reader
            .stepSplitter()   // Transformer
            .stepEmbedder()   // Transformer
            .stepWriter();    // Writer
    }

【核心概念】
  IngestionPipeline 的工作流程：
    Documents → [TransformComponent...] → Nodes → VectorStore

  TransformComponent 包括：
    - Reader（可选，如果输入已经是 Document）
    - NodeParser（切分）
    - EmbeddingGenerator（向量化）
    - VectorStore（存储）
    - 自定义 TransformComponent

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/08_IngestionPipeline摄取管线.py

【前置知识】
  第 1-7 课的所有知识
"""

import os
import shutil
from pathlib import Path
from llama_index.core import Settings


# ============================================================================
# 第 1 节：IngestionPipeline 基础
# ============================================================================

def demo_ingestion_pipeline_basic():
    """
    IngestionPipeline 是最简洁的数据摄入方式

    它把 Reader → Splitter → Embedder → VectorStore 串成一条线。
    """
    print("=" * 60)
    print("【IngestionPipeline 基础】")
    print("=" * 60)

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import Document

    # 准备文档
    documents = [
        Document(
            text="公司实行每日八小时工作制。上午9:00-12:00，下午13:30-17:30。",
            metadata={"file": "考勤制度.txt"}
        ),
        Document(
            text="员工每年享有5天带薪年假，满10年10天，满20年15天。",
            metadata={"file": "年假制度.txt"}
        ),
        Document(
            text="迟到早退超过30分钟按旷工半天处理。每月有3次豁免。",
            metadata={"file": "考勤制度.txt"}
        ),
    ]

    print("\n  --- 创建 Pipeline ---")
    # 第 1 步：定义 Pipeline 的组件
    pipeline = IngestionPipeline(
        transformations=[          # 转换链：按顺序执行
            TokenTextSplitter(     # 1. 切分文档
                chunk_size=256,
                chunk_overlap=20
            ),
            # 2. 向量化（自动使用 Settings.embed_model）
            # 3. 存储（自动使用 SimpleVectorStore）
        ],
        vector_store=SimpleVectorStore()  # 向量存储
    )

    print("    ✓ Pipeline 创建完成")
    print("    转换链: TokenTextSplitter → Embedding → VectorStore")

    # 第 2 步：运行 Pipeline
    print("\n  --- 运行 Pipeline ---")
    nodes = pipeline.run(documents=documents)
    print(f"    ✓ 处理完成，生成了 {len(nodes)} 个 Node")

    # 第 3 步：查看结果
    print("\n  --- 生成的 Node ---")
    for i, node in enumerate(nodes):
        print(f"    Node {i + 1}: {node.text[:50]}...")
        print(f"      元数据: {dict(node.metadata)}")


# ============================================================================
# 第 2 节：Pipeline 的 Transformations 链
# ============================================================================

def demo_transformations_chain():
    """
    transformations 是 Pipeline 的核心

    它是一个列表，每个元素都是一个 TransformComponent（转换组件）。
    数据按顺序通过每个组件，就像工厂流水线上的产品。

    类比 Java：
      List<Transformer> transformers = List.of(
          new Splitter(),
          new Embedder(),
          new Storer()
      );
      for (T doc : documents) {
          for (Transformer t : transformers) {
              doc = t.transform(doc);
          }
          results.add(doc);
      }
    """
    print("=" * 60)
    print("【Transformations 链详解】")
    print("=" * 60)

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import Document

    print("\n  --- 常见的 Transformation 组件 ---")
    print("""
  组件类型                    作用                    类比 Java
  ─────────────────────────────────────────────────────────────────
  TokenTextSplitter          按 token 数切分         String.split(maxTokens)
  SentenceSplitter           按句子切分              识别句号/问号/感叹号
  MarkdownNodeParser         按 Markdown 标题切分    解析 # ## ###
  HierarchicalNodeParser     创建层次化节点         树形结构
  SentenceWindowNodeParser   带上下文的句子切分      滑动窗口
  EmbeddingGenerator         生成向量嵌入            model.encode(text)
  MetadataFieldExtractor     提取元数据字段          BeanUtils.describe()
  AutoMergingParser          自动合并小节点          合并相邻小块
  """)

    print("\n  --- 自定义 Transformations 链 ---")

    # 方式 1：只用 Splitter（不存储到 VectorStore）
    print("\n  方式 1：只切分，不存储")
    pipeline1 = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=256, chunk_overlap=20)
        ]
    )
    docs = [Document(text="这是一段测试文本。")]
    nodes1 = pipeline1.run(documents=docs)
    print(f"    输入 1 个 Document，输出 {len(nodes1)} 个 Node")

    # 方式 2：Splitter + VectorStore
    print("\n  方式 2：切分 + 存储到 VectorStore")
    pipeline2 = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=256, chunk_overlap=20),
        ],
        vector_store=SimpleVectorStore()
    )
    nodes2 = pipeline2.run(documents=docs)
    print(f"    输入 1 个 Document，输出 {len(nodes2)} 个 Node，存入 VectorStore")

    # 方式 3：多个 Splitter 串联
    print("\n  方式 3：多层级切分（先按句子，再按 token）")
    pipeline3 = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=256),       # 先按句子切
            TokenTextSplitter(chunk_size=128),       # 再按 token 切
        ],
        vector_store=SimpleVectorStore()
    )
    nodes3 = pipeline3.run(documents=docs)
    print(f"    双层切分后输出 {len(nodes3)} 个 Node")


# ============================================================================
# 第 3 节：Pipeline 的缓存机制
# ============================================================================

def demo_pipeline_caching():
    """
    IngestionPipeline 支持缓存，避免重复处理相同文档

    这是生产环境非常重要的功能：
    - 第一次：处理所有文档 → 缓存结果
    - 第二次：只处理新增/修改的文档 → 大幅提速

    类比 Java：
      // 没有缓存：每次都重新计算
      List<Node> nodes = process(documents);

      // 有缓存：相同输入直接返回缓存结果
      List<Node> nodes = cache.getOrCompute(documents, () -> process(documents));
    """
    print("=" * 60)
    print("【Pipeline 缓存机制】")
    print("=" * 60)

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import Document

    cache_dir = "./pipeline_cache"

    print("\n  --- 第 1 次运行（无缓存，需要处理）---")
    pipeline1 = IngestionPipeline(
        transformations=[TokenTextSplitter(chunk_size=256)],
        vector_store=SimpleVectorStore(),
        cache_dir=cache_dir  # 启用缓存
    )

    documents = [
        Document(text="公司实行每日八小时工作制。", metadata={"file": "考勤.txt"}),
        Document(text="员工每年享有5天年假。", metadata={"file": "年假.txt"}),
    ]

    nodes1 = pipeline1.run(documents=documents)
    print(f"    处理了 {len(documents)} 个文档，生成 {len(nodes1)} 个 Node")
    print(f"    缓存已保存到: {cache_dir}")

    print("\n  --- 第 2 次运行（有缓存，直接读取）---")
    pipeline2 = IngestionPipeline(
        transformations=[TokenTextSplitter(chunk_size=256)],
        vector_store=SimpleVectorStore(),
        cache_dir=cache_dir  # 相同的缓存目录
    )

    nodes2 = pipeline2.run(documents=documents)
    print(f"    从缓存读取了 {len(nodes2)} 个 Node（几乎瞬间完成！）")

    print("\n  --- 第 3 次运行（部分缓存命中）---")
    # 如果文档有变化，只会重新处理变化的部分
    documents_modified = [
        Document(text="公司实行每日八小时工作制。", metadata={"file": "考勤.txt"}),
        Document(text="员工每年享有5天年假。新增：婚假15天。", metadata={"file": "年假.txt"}),  # 修改了
    ]
    nodes3 = pipeline2.run(documents=documents_modified)
    print(f"    只有变化的文档被重新处理，其余从缓存读取")

    # 清理
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"\n    ✓ 已清理缓存目录")


# ============================================================================
# 第 4 节：Pipeline 与 Index 的结合
# ============================================================================

def demo_pipeline_with_index():
    """
    IngestionPipeline 处理后，可以直接用结果构建 Index

    这是最常见的用法：
    Pipeline.run() → List[Node] → VectorStoreIndex.from_nodes()
    """
    print("=" * 60)
    print("【Pipeline + Index 结合使用】")
    print("=" * 60)

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.schema import Document

    # 准备文档
    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
        Document(text="迟到超过30分钟按旷工处理。"),
    ]

    print("\n  --- 第 1 步：用 Pipeline 处理文档 ---")
    pipeline = IngestionPipeline(
        transformations=[TokenTextSplitter(chunk_size=256)],
        vector_store=SimpleVectorStore()
    )
    nodes = pipeline.run(documents=documents)
    print(f"    ✓ 生成了 {len(nodes)} 个 Node")

    print("\n  --- 第 2 步：用 Node 构建 Index ---")
    index = VectorStoreIndex(nodes)
    print(f"    ✓ Index 构建完成")

    print("\n  --- 第 3 步：用 Index 查询 ---")
    query_engine = index.as_query_engine()
    response = query_engine.query("年假几天？")
    print(f"    问: 年假几天？")
    print(f"    答: {response.response}")


# ============================================================================
# 第 5 节：Pipeline 的高级用法 — 自定义 TransformComponent
# ============================================================================

def demo_custom_transform_component():
    """
    你可以创建自定义的 TransformComponent，在 Pipeline 中插入自定义逻辑

    类比 Java：
      // 实现一个接口
      public class MyFilter implements TransformComponent {
          @Override
          public List[Node] run(List[Node] nodes) {
              // 自定义过滤逻辑
              return nodes.stream()
                  .filter(n -> n.getText().length() > 10)
                  .collect(toList());
          }
      }
    """
    print("=" * 60)
    print("【自定义 TransformComponent】")
    print("=" * 60)

    from llama_index.core.schema import TransformComponent, BaseNode
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import Document, TextNode

    # 定义自定义转换器
    class LengthFilter(TransformComponent):
        """
        过滤掉文本长度小于指定阈值的 Node
        """
        def __init__(self, min_length=10):
            self.min_length = min_length

        def __call__(self, nodes):
            """
            __call__ 方法是 TransformComponent 的核心

            输入：Node 列表
            输出：过滤后的 Node 列表
            """
            filtered = [
                node for node in nodes
                if len(node.text) >= self.min_length
            ]
            print(f"    [LengthFilter] 输入 {len(nodes)} 个，输出 {len(filtered)} 个")
            return filtered

    # 创建包含自定义组件的 Pipeline
    print("\n  --- 创建带自定义组件的 Pipeline ---")
    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=256),    # 1. 切分
            LengthFilter(min_length=5),            # 2. 自定义过滤
        ],
        vector_store=SimpleVectorStore()
    )

    documents = [
        Document(text="短的", metadata={"tag": "short"}),
        Document(text="这是一个足够长的文档，不会被过滤掉。", metadata={"tag": "long"}),
        Document(text="也很短", metadata={"tag": "short2"}),
    ]

    print("\n  --- 运行 Pipeline ---")
    nodes = pipeline.run(documents=documents)
    print(f"\n    最终输出 {len(nodes)} 个 Node:")
    for node in nodes:
        print(f"      - {node.text} (tag: {node.metadata.get('tag')})")


# ============================================================================
# 第 6 节：Pipeline vs 手动方式对比
# ============================================================================

def demo_pipeline_vs_manual():
    """
    对比 IngestionPipeline 和手动方式的代码量

    这能帮你直观理解 Pipeline 的价值。
    """
    print("=" * 60)
    print("【Pipeline vs 手动方式对比】")
    print("=" * 60)

    print("""
  ┌─────────────────────┬──────────────────────────┬──────────────────────────┐
  │ 步骤                │ 手动方式                  │ IngestionPipeline         │
  ├─────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 1. 读取文档          │ reader.load_data()       │ pipeline.run(documents)  │
  │ 2. 切分文档          │ splitter.split(docs)     │ (自动)                    │
  │ 3. 向量化           │ embed_model.encode()     │ (自动)                    │
  │ 4. 存入 VectorStore │ vector_store.add()       │ (自动)                    │
  │ 5. 构建 Index       │ VectorStoreIndex(...)    │ index = VectorStoreIndex  │
  │                     │                          │   .from_nodes(pipeline..) │
  ├─────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 代码行数             │ ~15 行                   │ ~3 行                     │
  │ 可维护性             │ 分散在各处               │ 集中在 Pipeline 定义      │
  │ 缓存支持             │ 无                       │ 内置缓存                  │
  │ 增量更新             │ 手动实现                 │ pipeline.refresh()        │
  └─────────────────────┴──────────────────────────┴──────────────────────────┘

  手动方式代码：
    reader = SimpleDirectoryReader(input_dir="./docs")
    documents = reader.load_data()

    splitter = TokenTextSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)

    vector_store = FAISSVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    for node in nodes:
        vector_store.add([node])

    index = VectorStoreIndex(nodes, storage_context=storage_context)

  IngestionPipeline 代码：
    pipeline = IngestionPipeline(
        transformations=[TokenTextSplitter(chunk_size=512)],
        vector_store=FAISSVectorStore(faiss_index)
    )
    nodes = pipeline.run(documents=documents)
    index = VectorStoreIndex(nodes)
    """)


# ============================================================================
# 第 7 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 8 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. IngestionPipeline 把 Reader → Splitter → Embed → Store 串成一条线   │
│  2. transformations 参数定义处理链的顺序                                 │
│  3. 支持缓存，避免重复处理                                              │
│  4. 可以自定义 TransformComponent 插入个性化逻辑                        │
│  5. 一行代码完成整个数据摄入流程                                        │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 基本用法                                                             │
│  from llama_index.core.ingestion import IngestionPipeline               │
│  from llama_index.core.node_parser import TokenTextSplitter             │
│                                                                       │
│  pipeline = IngestionPipeline(                                          │
│      transformations=[                                                  │
│          TokenTextSplitter(chunk_size=512),                             │
│      ],                                                                 │
│      vector_store=my_vector_store                                       │
│  )                                                                      │
│                                                                         │
│  nodes = pipeline.run(documents=documents)                              │
│                                                                         │
│  # 带缓存                                                               │
│  pipeline = IngestionPipeline(                                          │
│      transformations=[...],                                             │
│      cache_dir="./cache"                                                │
│  )                                                                      │
│                                                                         │
│  # 自定义 TransformComponent                                           │
│  class MyFilter(TransformComponent):                                    │
│      def __call__(self, nodes):                                         │
│          return [n for n in nodes if len(n.text) > 10]                  │
│                                                                         │
│  pipeline = IngestionPipeline(                                          │
│      transformations=[MyFilter(), ...]                                  │
│  )                                                                      │
│                                                                         │
│  【下一课预告】                                                           │
│  第 9 课：Chat Engine + Memory / 对话引擎与记忆                          │
│  类比 Java：HttpSession + 对话状态管理                                   │
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
║   第 8 课：Ingestion Pipeline / 摄取管线 — 一键完成数据  ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. IngestionPipeline 基础                              ║
║   2. Transformations 链详解                              ║
║   3. Pipeline 缓存机制                                   ║
║   4. Pipeline + Index 结合                               ║
║   5. 自定义 TransformComponent                           ║
║   6. Pipeline vs 手动方式对比                            ║
║   7. 总结                                               ║
║                                                          ║
║   前置知识：第 1-7 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：IngestionPipeline 基础 ━━━")
    demo_ingestion_pipeline_basic()

    print("\n━━━ 第 2 节：Transformations 链 ━━━")
    demo_transformations_chain()

    print("\n━━━ 第 3 节：Pipeline 缓存 ━━━")
    demo_pipeline_caching()

    print("\n━━━ 第 4 节：Pipeline + Index ━━━")
    demo_pipeline_with_index()

    print("\n━━━ 第 5 节：自定义 TransformComponent ━━━")
    demo_custom_transform_component()

    print("\n━━━ 第 6 节：Pipeline vs 手动方式 ━━━")
    demo_pipeline_vs_manual()

    print("\n🎉 第 8 课完成！")
    print("   建议下一步：阅读 week03/code/p30-ingestion-pipeline.ipynb")
