"""
===============================================================================
 第 11 课：综合项目 — 完整 RAG 系统
===============================================================================

【这一课学什么？】
  终于到了最后一课！我们将前面 10 课的所有知识串起来，
  构建一个完整的 RAG（检索增强生成）系统。

  这个项目将包含：
    1. 文档上传（Reader）
    2. 文档切分（NodeParser）
    3. 向量化存储（VectorStore）
    4. 索引构建（Index）
    5. 检索与过滤（Retriever + Postprocessor）
    6. 查询回答（QueryEngine）
    7. 多轮对话（ChatEngine + Memory）
    8. 工具调用（Agent + Tools）

【类比 Java】
  这是你的"毕业设计"——把 MVC 三层架构的所有知识串起来做一个完整应用。
  - Controller → ChatEngine（接收用户请求）
  - Service → QueryEngine（业务逻辑）
  - Repository → VectorStore（数据访问）
  - Config → Settings（全局配置）

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/11_综合项目完整RAG.py

【前置知识】
  第 1-10 课的所有知识
"""

import os
import shutil
from pathlib import Path
from llama_index.core import Settings


# ============================================================================
# 第 1 节：项目架构设计
# ============================================================================

def demo_architecture():
    """
    展示整个 RAG 系统的架构

    这个架构图帮助你理解各个组件之间的关系。
    """
    print("=" * 60)
    print("【RAG 系统架构】")
    print("=" * 60)

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                    完整 RAG 系统架构                         │
  │                                                             │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
  │  │  User    │───→│ Chat     │───→│ Query    │              │
  │  │  Input   │    │ Engine   │    │ Engine   │              │
  │  └──────────┘    └────┬─────┘    └────┬─────┘              │
  │                       │               │                     │
  │              ┌────────▼──────┐  ┌─────▼────────┐           │
  │              │  Memory       │  │  Post-       │           │
  │              │  (对话记忆)    │  │  processor  │           │
  │              └───────────────┘  └─────┬────────┘           │
  │                                       │                     │
  │              ┌────────────────────────▼──────────┐         │
  │              │         Retriever                 │         │
  │              │   (向量相似度搜索 + 关键词过滤)    │         │
  │              └────────────────────────┬──────────┘         │
  │                                       │                     │
  │              ┌────────────────────────▼──────────┐         │
  │              │         VectorStoreIndex           │         │
  │              │   (Index + VectorStore + DocStore) │         │
  │              └────────────────────────┬──────────┘         │
  │                                       │                     │
  │              ┌────────────────────────▼──────────┐         │
  │              │      IngestionPipeline             │         │
  │              │  (Reader → Splitter → Embed → Store)│        │
  │              └────────────────────────┬──────────┘         │
  │                                       │                     │
  │              ┌────────────────────────▼──────────┐         │
  │              │        Document Store              │         │
  │              │  (原始文档 / PDF / TXT / DOCX)     │         │
  │              └───────────────────────────────────┘         │
  │                                                             │
  │  配置层: Settings (LLM + Embedding + ChunkSize)             │
  │  工具层: Agent + Tools (计算器 / 数据库 / API)              │
  └─────────────────────────────────────────────────────────────┘

  数据流向：
    文档上传 → Reader → Document → Splitter → Node → Embed → VectorStore
    用户提问 → ChatEngine → Retriever → Node → Postprocessor → LLM → 回答
    """)


# ============================================================================
# 第 2 节：Step 1 — 配置 Settings
# ============================================================================

def step1_configure_settings():
    """
    第一步：配置全局 Settings

    这是所有后续操作的基础。
    """
    print("=" * 60)
    print("【Step 1: 配置 Settings】")
    print("=" * 60)

    from llama_index.llms.dashscope import DashScope
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("  ⚠️  缺少 DASHSCOPE_API_KEY，使用模拟模式")
        print("  请设置: export DASHSCOPE_API_KEY='your-key'")
        return False

    # 配置 LLM
    Settings.llm = DashScope(
        model_name="qwen-plus",
        api_key=api_key,
        temperature=0.1,
    )
    print(f"  ✓ LLM: {Settings.llm.model_name}")

    # 配置 Embedding
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=api_key,
        embed_batch_size=10,
    )
    print(f"  ✓ Embedding: {Settings.embed_model.model_name}")

    # 配置分块参数
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    Settings.num_output = 256
    Settings.context_window = 128000

    print(f"  ✓ chunk_size: {Settings.chunk_size}")
    print(f"  ✓ chunk_overlap: {Settings.chunk_overlap}")
    print(f"  ✓ num_output: {Settings.num_output}")

    return True


# ============================================================================
# 第 3 节：Step 2 — 准备文档
# ============================================================================

def step2_prepare_documents():
    """
    第二步：准备文档（模拟 Reader 读取的结果）

    实际项目中，这里会用 SimpleDirectoryReader 从文件夹读取。
    """
    print("\n" + "=" * 60)
    print("【Step 2: 准备文档】")
    print("=" * 60)

    from llama_index.core import Document

    # 创建测试文档（模拟从 PDF/TXT 读取的内容）
    documents = [
        Document(
            text="公司实行每日八小时工作制。上午工作时间为9:00至12:00，下午工作时间为13:30至17:30。中午休息1小时。",
            metadata={"file": "考勤制度.pdf", "page": 1, "category": "考勤"}
        ),
        Document(
            text="员工每周享有两天休息日，通常为周六和周日。因工作需要安排加班的，应优先安排补休。无法安排补休的，按国家规定支付加班工资。",
            metadata={"file": "考勤制度.pdf", "page": 2, "category": "考勤"}
        ),
        Document(
            text="员工每年享有5天带薪年假。工作满10年不满20年的，年休假10天。满20年及以上的，年休假15天。",
            metadata={"file": "年假制度.pdf", "page": 1, "category": "假期"}
        ),
        Document(
            text="病假需提供二级以上医院诊断证明。病假30天以内按基本工资80%发放，30天以上按基本工资60%发放。",
            metadata={"file": "病假制度.pdf", "page": 1, "category": "假期"}
        ),
        Document(
            text="婚假15天，产假98天，陪产假15天。法定节假日共11天，包括元旦1天、春节3天、国庆节3天等。",
            metadata={"file": "假期汇总.pdf", "page": 1, "category": "假期"}
        ),
        Document(
            text="员工入职即缴纳五险一金。公积金比例个人12%公司12%。社保按缴费基数8%个人承担，16%公司承担。",
            metadata={"file": "福利制度.pdf", "page": 1, "category": "福利"}
        ),
        Document(
            text="年终奖金根据个人绩效评定。A级（优秀）6个月工资，B级（良好）3个月工资，C级（合格）1个月工资，D级（不合格）无奖金。",
            metadata={"file": "薪酬制度.pdf", "page": 1, "category": "薪酬"}
        ),
        Document(
            text="员工月薪于次月15日发放。迟到早退超过30分钟按旷工半天处理。每月有3次迟到豁免机会。",
            metadata={"file": "考勤制度.pdf", "page": 3, "category": "考勤"}
        ),
    ]

    print(f"  ✓ 准备了 {len(documents)} 个文档")
    print(f"  类别分布:")

    # 统计类别
    categories = {}
    for doc in documents:
        cat = doc.metadata.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in categories.items():
        print(f"    {cat}: {count} 个文档")

    return documents


# ============================================================================
# 第 4 节：Step 3 — 构建 Index
# ============================================================================

def step3_build_index(documents):
    """
    第三步：构建 VectorStoreIndex

    这一步自动完成：切分 → 向量化 → 存储。
    """
    print("\n" + "=" * 60)
    print("【Step 3: 构建 Index】")
    print("=" * 60)

    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 一行代码构建 Index
    # 内部自动：
    #   1. 用 TokenTextSplitter 切分文档
    #   2. 用 Settings.embed_model 向量化
    #   3. 存入内存 VectorStore
    index = VectorStoreIndex.from_documents(documents)

    print(f"  ✓ Index 构建完成")
    print(f"  文档数: {len(list(index.docstore.docs.values()))}")

    return index


# ============================================================================
# 第 5 节：Step 4 — 创建查询引擎
# ============================================================================

def step4_create_query_engine(index):
    """
    第四步：创建 QueryEngine

    这里我们加入 Postprocessor 来提升检索质量。
    """
    print("\n" + "=" * 60)
    print("【Step 4: 创建查询引擎】")
    print("=" * 60)

    from llama_index.core.postprocessor import SimilarityPostprocessor

    # 创建带 Postprocessor 的 QueryEngine
    query_engine = index.as_query_engine(
        similarity_top_k=3,               # 检索前 3 个最相似节点
        response_mode="compact",          # 紧凑模式（快速）
        streaming=False,                  # 非流式
        postprocessors=[                  # 添加后处理器
            SimilarityPostprocessor(      # 过滤低相似度节点
                similarity_cutoff=0.01
            )
        ],
    )

    print("  ✓ QueryEngine 创建完成")
    print("    - similarity_top_k: 3")
    print("    - response_mode: compact")
    print("    - postprocessor: SimilarityPostprocessor (cutoff=0.01)")

    return query_engine


# ============================================================================
# 第 6 节：Step 5 — 测试查询
# ============================================================================

def step5_test_query_engine(query_engine):
    """
    第五步：测试查询引擎

    用各种问题测试 RAG 系统的回答质量。
    """
    print("\n" + "=" * 60)
    print("【Step 5: 测试查询引擎】")
    print("=" * 60)

    questions = [
        "员工每天工作几个小时？",
        "年假有多少天？",
        "迟到怎么处罚？",
        "五险一金怎么交？",
        "年终奖怎么算？",
        "病假需要什么材料？",
    ]

    print("\n  --- 查询测试 ---\n")

    for i, question in enumerate(questions, 1):
        print(f"  [{i}] 问: {question}")
        response = query_engine.query(question)
        print(f"      答: {response.response}")
        print(f"      参考了 {len(response.source_nodes)} 个文档")
        print()


# ============================================================================
# 第 7 节：Step 6 — 创建对话引擎
# ============================================================================

def step6_create_chat_engine(index):
    """
    第六步：创建 ChatEngine

    支持多轮对话，记住上下文。
    """
    print("\n" + "=" * 60)
    print("【Step 6: 创建对话引擎】")
    print("=" * 60)

    from llama_index.core.chat_engine import ConversableQueryEngineChatEngine

    chat_engine = ConversableQueryEngineChatEngine(
        index=index,
        token_limit=4096,
    )

    print("  ✓ ChatEngine 创建完成")

    # 模拟多轮对话
    print("\n  --- 多轮对话测试 ---\n")

    conversations = [
        ("用户", "年假几天？"),
        ("AI", ""),  # 等待回答
        ("用户", "那满10年呢？"),  # 引用上文
        ("AI", ""),
        ("用户", "婚假呢？"),  # 切换话题
        ("AI", ""),
    ]

    for speaker, question in conversations:
        if speaker == "用户":
            print(f"  {speaker}: {question}")
            response = chat_engine.chat(question)
            conversations[conversations.index((speaker, question)) + 1] = ("AI", response.response)
        else:
            print(f"  {speaker}: {question[:60]}...")

    print()


# ============================================================================
# 第 8 节：Step 7 — 创建 Agent
# ============================================================================

def step7_create_agent(index):
    """
    第七步：创建 Agent

    让 LLM 可以自主选择使用知识库查询还是其他工具。
    """
    print("\n" + "=" * 60)
    print("【Step 7: 创建 Agent】")
    print("=" * 60)

    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool, QueryEngineTool

    # 定义工具
    def add(a: int, b: int) -> int:
        """计算两个数的和"""
        return a + b

    def multiply(a: int, b: int) -> int:
        """计算两个数的乘积"""
        return a * b

    # 创建工具
    add_tool = FunctionTool.from_defaults(add, name="calculator_add", description="加法计算器")
    mul_tool = FunctionTool.from_defaults(multiply, name="calculator_mul", description="乘法计算器")

    # 把 QueryEngine 也包装成工具
    knowledge_tool = QueryEngineTool(
        query_engine=index.as_query_engine(),
        name="knowledge_base",
        description="查询公司制度（年假、病假、考勤、福利等）"
    )

    # 创建 Agent
    agent = ReActAgent.from_tools(
        tools=[add_tool, mul_tool, knowledge_tool],
        max_iterations=5,
    )

    print("  ✓ Agent 创建完成")
    print("    可用工具: calculator_add, calculator_mul, knowledge_base")

    # 测试：Agent 需要同时使用计算器和知识库
    print("\n  --- Agent 测试 ---\n")

    test_questions = [
        "5 + 3 等于多少？",
        "年假几天？",
        "10乘以5再加3等于多少？",  # 需要连续调用多个工具
    ]

    for question in test_questions:
        print(f"  问: {question}")
        response = agent.chat(question)
        print(f"  答: {response.response[:80]}...")
        print()


# ============================================================================
# 第 9 节：Step 8 — 使用 IngestionPipeline 重新构建
# ============================================================================

def step8_use_pipeline():
    """
    第八步：使用 IngestionPipeline 重新构建整个系统

    这是最简洁的方式，一行代码完成所有步骤。
    """
    print("\n" + "=" * 60)
    print("【Step 8: 使用 IngestionPipeline 一键构建】")
    print("=" * 60)

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.schema import Document

    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
        Document(text="迟到超过30分钟按旷工处理。"),
    ]

    # 一行代码：读取 → 切分 → 向量化 → 存储
    pipeline = IngestionPipeline(
        transformations=[TokenTextSplitter(chunk_size=256)],
        vector_store=SimpleVectorStore()
    )

    nodes = pipeline.run(documents=documents)
    print(f"  ✓ Pipeline 运行完成，生成 {len(nodes)} 个 Node")

    # 用 Pipeline 的输出构建 Index
    from llama_index.core.indices.vector_store import VectorStoreIndex
    index = VectorStoreIndex(nodes)
    print(f"  ✓ Index 构建完成")


# ============================================================================
# 第 10 节：完整项目代码（可复用的模板）
# ============================================================================

def demo_complete_project():
    """
    完整可复用的 RAG 项目模板

    你可以把这个函数复制到自己的项目中，修改文档路径即可使用。
    """
    print("=" * 60)
    print("【完整 RAG 项目模板】")
    print("=" * 60)

    print("""
  # 完整 RAG 系统模板（复制即用）
  # ========================================

  from llama_index.core import Settings, Document
  from llama_index.core.indices.vector_store import VectorStoreIndex
  from llama_index.core.chat_engine import ConversableQueryEngineChatEngine
  from llama_index.llms.dashscope import DashScope
  from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
  import os

  # 1. 配置 Settings
  Settings.llm = DashScope(
      model_name="qwen-plus",
      api_key=os.environ["DASHSCOPE_API_KEY"],
      temperature=0.1,
  )
  Settings.embed_model = DashScopeEmbedding(
      model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
      api_key=os.environ["DASHSCOPE_API_KEY"],
  )

  # 2. 加载文档（替换为你的文件路径）
  documents = [
      Document(text="你的文档内容...", metadata={"file": "文件名"})
  ]
  # 或者用 SimpleDirectoryReader 从文件夹读取:
  # from llama_index.core.readers.simple_file_based import SimpleDirectoryReader
  # reader = SimpleDirectoryReader(input_dir="./docs")
  # documents = reader.load_data()

  # 3. 构建 Index
  index = VectorStoreIndex.from_documents(documents)

  # 4. 创建 ChatEngine
  chat_engine = ConversableQueryEngineChatEngine(index=index)

  # 5. 开始对话
  while True:
      question = input("你: ")
      if question in ["quit", "exit", "q"]:
          break
      response = chat_engine.chat(question)
      print(f"AI: {response.response}")
  """)


# ============================================================================
# 第 11 节：项目总结与下一步
# ============================================================================

def demo_summary():
    """
    总结整个学习旅程，规划下一步
    """
    print("\n" + "=" * 60)
    print("【学习总结与下一步】")
    print("=" * 60)

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                    学习路线图回顾                             │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  ✅ 第 1 课: Node / 文本切片                                │
  │  ✅ 第 2 课: Settings / 全局配置                            │
  │  ✅ 第 3 课: Reader / 文档加载                              │
  │  ✅ 第 4 课: Vector Store / 向量存储                        │
  │  ✅ 第 5 课: Index + Retriever / 索引与检索                  │
  │  ✅ 第 6 课: Postprocessor / 重排序                         │
  │  ✅ 第 7 课: Query Engine / 查询引擎                        │
  │  ✅ 第 8 课: Ingestion Pipeline / 摄取管线                  │
  │  ✅ 第 9 课: Chat Engine + Memory / 对话引擎与记忆           │
  │  ✅ 第 10 课: Agent + Tools / 智能体与工具                  │
  │  ✅ 第 11 课: 综合项目 — 完整 RAG 系统                      │
  │                                                             │
  │  恭喜！你已经掌握了 LlamaIndex 的核心知识！                  │
  │                                                             │
  │  【下一步建议】                                              │
  │  1. 阅读 week03-local-rag/ 的完整 RAG 项目源码               │
  │  2. 尝试用自己的文档构建 RAG 系统                            │
  │  3. 学习高级主题：                                          │
  │     - KnowledgeGraphIndex（知识图谱）                       │
  │     - MultiModalVectorStoreIndex（多模态）                  │
  │     - LlamaIndex Evaluation（评估）                         │
  │     - LlamaIndex Workflow（工作流）                         │
  │  4. 部署到生产环境：                                        │
  │     - FastAPI 封装为 REST API                               │
  │     - Docker 容器化部署                                     │
  │     - Milvus / Elasticsearch 替代 FAISS                     │
  │                                                             │
  │  【Python 进阶建议】                                         │
  │  1. 学习 async/await（异步编程）                            │
  │  2. 学习 FastAPI（Web 框架）                                │
  │  3. 学习 Docker（容器化）                                   │
  │  4. 学习 pytest（测试）                                     │
  │  5. 学习 CI/CD（GitHub Actions）                            │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
    """)


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   第 11 课：综合项目 — 完整 RAG 系统                      ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. 项目架构设计                                        ║
║   2. Step 1: 配置 Settings                               ║
║   3. Step 2: 准备文档                                    ║
║   4. Step 3: 构建 Index                                  ║
║   5. Step 4: 创建查询引擎                                ║
║   6. Step 5: 测试查询                                    ║
║   7. Step 6: 创建对话引擎                                ║
║   8. Step 7: 创建 Agent                                  ║
║   9. Step 8: 使用 IngestionPipeline                      ║
║   10. 完整项目模板                                       ║
║   11. 总结与下一步                                       ║
║                                                          ║
║   前置知识：第 1-10 课                                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：项目架构设计 ━━━")
    demo_architecture()

    print("\n━━━ 第 2 节：配置 Settings ━━━")
    step1_configure_settings()

    print("\n━━━ 第 3 节：准备文档 ━━━")
    documents = step2_prepare_documents()

    if documents:
        print("\n━━━ 第 4 节：构建 Index ━━━")
        index = step3_build_index(documents)

        print("\n━━━ 第 5 节：创建查询引擎 ━━━")
        query_engine = step4_create_query_engine(index)

        print("\n━━━ 第 6 节：测试查询 ━━━")
        step5_test_query_engine(query_engine)

        print("\n━━━ 第 7 节：创建对话引擎 ━━━")
        step6_create_chat_engine(index)

        print("\n━━━ 第 8 节：创建 Agent ━━━")
        step7_create_agent(index)

    print("\n━━━ 第 9 节：使用 IngestionPipeline ━━━")
    step8_use_pipeline()

    print("\n━━━ 第 10 节：完整项目模板 ━━━")
    demo_complete_project()

    print("\n━━━ 第 11 节：总结与下一步 ━━━")
    demo_summary()

    print("\n🎉🎉🎉 全部 11 课完成！🎉🎉🎉")
    print("   恭喜你完成了 LlamaIndex + Python 的系统学习！")
    print("   建议下一步：")
    print("   1. 阅读 week03-local-rag/ 的完整项目源码")
    print("   2. 用自己的文档构建一个 RAG 系统")
    print("   3. 继续学习 LangChain（第 4 课）")
