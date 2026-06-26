"""
===============================================================================
 第 7 课：Query Engine / 查询引擎 — 生成最终回答
===============================================================================

【这一课学什么？】
  第 5 课我们学会了用 Retriever 检索相关文档，第 6 课学会了用 Postprocessor
  过滤和重排序。但检索到的文档还不是最终答案——我们需要把它们交给 LLM，
  让 LLM 基于检索到的信息生成自然语言回答。
  Query Engine 就是做这件事的。

【类比 Java】
  Query Engine ≈ Service 层
  - Retriever 返回的是"原材料"（Node 列表）
  - Query Engine 负责：检索 → 组装 Prompt → 调用 LLM → 生成回答
  - 类比：Controller 调用 Service，Service 组装数据返回给前端

【核心概念】
  Query Engine 的工作流程：
    用户问题 → QueryEngine → (Retriever → Postprocessor → LLM) → 回答

  最关键的方法：
    query_engine = index.as_query_engine()
    response = query_engine.query("用户的问题")
    print(response.response)  # LLM 生成的回答
    print(response.source_nodes)  # 用来生成回答的源节点

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/07_QueryEngine查询引擎.py

【前置知识】
  - 第 1 课：Node / 文本切片
  - 第 2 课：Settings / 全局配置
  - 第 5 课：Index + Retriever
  - 第 6 课：Postprocessor
"""

import os
from llama_index.core import Settings


# ============================================================================
# 第 1 节：as_query_engine() — 一行创建查询引擎
# ============================================================================

def demo_as_query_engine():
    """
    index.as_query_engine() 是 LlamaIndex 最核心的 API 之一

    它把 Index 变成了一个可以回答问题的查询引擎。
    内部自动完成：
    1. 用 Retriever 检索相关文档
    2. 把文档文本 + 用户问题组装成 Prompt
    3. 调用 Settings.llm 生成回答
    4. 返回 Response 对象

    类比 Java：
      // 手动方式（太麻烦）
      List<Node> nodes = retriever.retrieve(query);
      String context = nodes.stream().map(Node::getText).collect(joining("\n"));
      String prompt = "基于以下信息回答问题：\n" + context + "\n问题：" + query;
      Response response = llm.complete(prompt);

      // 一行搞定
      Response response = queryEngine.query(query);
    """
    print("=" * 60)
    print("【as_query_engine() 一行创建查询引擎】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 准备文档
    documents = [
        Document(text="公司实行每日八小时工作制。上午9:00-12:00，下午13:30-17:30。"),
        Document(text="员工每年享有5天带薪年假，满10年10天，满20年15天。"),
        Document(text="迟到早退超过30分钟按旷工半天处理。每月3次豁免。"),
        Document(text="入职即缴纳五险一金，公积金个人12%公司12%。"),
    ]

    # 创建 Index
    print("\n  --- 第 1 步：创建 Index ---")
    index = VectorStoreIndex.from_documents(documents)
    print("    ✓ Index 创建完成")

    # 创建 QueryEngine
    print("\n  --- 第 2 步：创建 QueryEngine ---")
    query_engine = index.as_query_engine()
    print("    ✓ QueryEngine 创建完成")

    # 执行查询
    print("\n  --- 第 3 步：执行查询 ---")
    questions = [
        "员工每天工作几个小时？",
        "年假有多少天？",
        "迟到怎么处罚？",
        "五险一金怎么交？",
    ]

    for question in questions:
        print(f"\n    问: {question}")
        response = query_engine.query(question)
        print(f"    答: {response.response}")
        print(f"    使用了 {len(response.source_nodes)} 个参考文档")


# ============================================================================
# 第 2 节：Response 对象详解
# ============================================================================

def demo_response_object():
    """
    query_engine.query() 返回一个 Response 对象

    Response 包含：
    - response       → LLM 生成的回答文本
    - source_nodes   → 用来生成回答的源节点列表
    - metadata       → 额外元数据（模型名、耗时等）
    - additional_info → 附加信息

    类比 Java：
      public class Response {
          private String response;        // 回答
          private List<NodeWithScore> sourceNodes;  // 来源
          private Map<String, Object> metadata;  // 元数据
      }
    """
    print("=" * 60)
    print("【Response 对象详解】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query("年假几天？")

    print("\n  --- Response 对象属性 ---")
    print(f"    类型: {type(response).__name__}")
    print(f"    .response          = {response.response}")
    print(f"    .source_nodes      = {len(response.source_nodes)} 个节点")
    print(f"    .metadata          = {response.metadata}")
    print(f"    .additional_info   = {response.additional_info}")

    print("\n  --- 源节点详情 ---")
    for i, node_score in enumerate(response.source_nodes):
        print(f"\n    源节点 {i + 1}:")
        print(f"      文本: {node_score.node.text}")
        print(f"      相似度: {node_score.score:.4f}")
        print(f"      元数据: {dict(node_score.node.metadata)}")


# ============================================================================
# 第 3 节：自定义 Prompt — 控制 LLM 的回答风格
# ============================================================================

def demo_custom_prompt():
    """
    默认情况下，QueryEngine 使用内置的 Prompt 模板。
    你可以自定义 Prompt，来控制 LLM 的回答风格。

    类比 Java：
      // 自定义 SQL 查询
      String sql = "SELECT * FROM employees WHERE dept = ?";
      PreparedStatement ps = conn.prepareStatement(sql);
      ps.setString(1, "技术部");

      // 自定义 Prompt
      String prompt = "你是一个专业的HR助手。请基于以下信息回答问题：\n{context}\n问题：{query}";
    """
    print("=" * 60)
    print("【自定义 Prompt — 控制回答风格】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.prompts import PromptTemplate

    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # 查看默认 Prompt
    print("\n  --- 默认 Prompt 模板 ---")
    default_prompt = index.as_query_engine().prompt
    print(f"    {default_prompt.template}")

    # 自定义 Prompt
    print("\n  --- 自定义 Prompt ---")

    # 方式 1：创建带自定义 prompt 的 QueryEngine
    custom_prompt = """你是一个专业的HR助手。请用简洁、友好的语气回答问题。
如果信息不足以回答问题，请说"抱歉，我没有找到相关信息"。

参考信息：
{context_str}

问题：{query_str}

回答："""

    query_engine = index.as_query_engine(
        text_qa_template=PromptTemplate(custom_prompt)
    )

    response = query_engine.query("年假几天？")
    print(f"    问: 年假几天？")
    print(f"    答: {response.response}")

    # 方式 2：更复杂的自定义 Prompt（带格式要求）
    print("\n  --- 更复杂的自定义 Prompt ---")
    formatted_prompt = """请基于以下参考信息回答问题。回答格式：

【答案】
（直接回答问题的句子）

【依据】
（引用相关文档的标题和内容）

参考信息：
{context_str}

问题：{query_str}

回答："""

    query_engine2 = index.as_query_engine(
        text_qa_template=PromptTemplate(formatted_prompt)
    )

    response2 = query_engine2.query("年假几天？")
    print(f"    问: 年假几天？")
    print(f"    答:\n{response2.response}")


# ============================================================================
# 第 4 节：QueryEngine 的参数详解
# ============================================================================

def demo_query_engine_params():
    """
    as_query_engine() 支持大量参数，控制检索和生成的行为

    这些参数就像 Java 方法的可选参数，用命名参数传递。
    """
    print("=" * 60)
    print("【QueryEngine 参数详解】")
    print("=" * 60)

    print("""
  as_query_engine() 常用参数：

  ┌──────────────────────┬──────────┬──────────────────────────────────┐
  │ 参数名               │ 类型     │ 说明                              │
  ├──────────────────────┼──────────┼──────────────────────────────────┤
  │ similarity_top_k     │ int      │ 检索返回前 k 个节点（默认 2）      │
  │ response_mode        │ str      │ 回答模式（见下方）                 │
  │ text_qa_template     │ Prompt   │ 问答 Prompt 模板                  │
  │ refine_template      │ Prompt   │ 精炼回答模板（refine 模式用）      │
  │ pre_filters          │ Filter   │ 元数据过滤条件                    │
  │ postprocessors       │ list     │ 后处理器列表                      │
  │ llm                  │ LLM      │ 指定使用的 LLM（覆盖 Settings）    │
  │ embed_model          │ Embedding│ 指定使用的 Embedding（覆盖 Settings）│
  │ streaming            │ bool     │ 是否流式输出（默认 False）          │
  └──────────────────────┴──────────┴──────────────────────────────────┘

  response_mode（回答模式）：
    - "compact"   → 紧凑模式：先合并所有节点文本，再一次性问 LLM（最快）
    - "refine"    → 精炼模式：逐个节点问 LLM，逐步完善回答（最准确但最慢）
    - "no_text"   → 只返回检索结果，不调用 LLM
    - "accumulate"→ 累积模式：每个节点的答案累加
    """)


# ============================================================================
# 第 5 节：不同 response_mode 的对比
# ============================================================================

def demo_response_modes():
    """
    演示不同 response_mode 的效果差异
    """
    print("=" * 60)
    print("【不同 response_mode 对比】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    documents = [
        Document(text="公司实行每日八小时工作制。上午9:00-12:00，下午13:30-17:30。"),
        Document(text="员工每年享有5天带薪年假，满10年10天，满20年15天。"),
        Document(text="迟到早退超过30分钟按旷工半天处理。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    print("\n  --- compact 模式（默认，最快）---")
    print("    原理：把所有检索到的节点文本拼成一个大字符串，一次性问 LLM")
    print("    类比 Java: String context = nodes.stream().map(n -> n.text).collect(joining(\"\\n\"));")
    print("               llm.complete(\"基于以下信息：\" + context + \"回答问题：\" + query);")

    engine_compact = index.as_query_engine(response_mode="compact")
    response = engine_compact.query("工作时间和年假规定是什么？")
    print(f"    回答: {response.response[:80]}...")

    print("\n  --- refine 模式（最准确，最慢）---")
    print("    原理：对每个节点分别问 LLM，然后把答案逐步精炼")
    print("    类比 Java: 对每个 node 调用 llm.complete(), 然后合并结果")

    engine_refine = index.as_query_engine(response_mode="refine")
    response = engine_refine.query("工作时间和年假规定是什么？")
    print(f"    回答: {response.response[:80]}...")

    print("""
  选择建议：
    - 节点少（< 5个）→ refine（更准确）
    - 节点多（> 5个）→ compact（更快）
    - 不确定 → 用默认 compact
    """)


# ============================================================================
# 第 6 节：流式输出（Streaming）
# ============================================================================

def demo_streaming():
    """
    流式输出让 LLM 的回答像打字一样逐字显示

    类比 Java：
      // 非流式：等 LLM 全部生成完再返回
      Response r = llm.complete(query);
      System.out.println(r.getText());

      // 流式：边生成边输出
      for (Token t : llm.stream(query)) {
          System.out.print(t.getText());
      }
    """
    print("=" * 60)
    print("【流式输出（Streaming）】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # 创建流式 QueryEngine
    print("\n  --- 非流式输出 ---")
    engine_normal = index.as_query_engine(streaming=False)
    response = engine_normal.query("年假几天？")
    print(f"    完整回答: {response.response}")

    print("\n  --- 流式输出 ---")
    engine_stream = index.as_query_engine(streaming=True)
    print("    逐字输出:")
    response_stream = engine_stream.query("年假几天？")
    # 流式输出会在终端逐字显示
    for delta in response_stream.response_gen:
        print(delta, end="", flush=True)
    print()  # 换行


# ============================================================================
# 第 7 节：完整示例 — 从文档到回答的全流程
# ============================================================================

def demo_complete_example():
    """
    完整示例：把前面学的所有知识串起来

    流程：
    1. 配置 Settings（第 2 课）
    2. 创建/读取文档（第 3 课）
    3. 构建 Index（第 5 课）
    4. 添加 Postprocessor（第 6 课）
    5. 创建 QueryEngine 并查询（第 7 课）
    """
    print("=" * 60)
    print("【完整示例：从文档到回答的全流程】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.schema import TextNode

    # 第 1 步：准备文档（模拟 Reader 读取的结果）
    print("\n  --- 第 1 步：准备文档 ---")
    documents = [
        Document(
            text="公司实行每日八小时工作制。上午9:00-12:00，下午13:30-17:30。",
            metadata={"file": "考勤制度.pdf", "page": 1}
        ),
        Document(
            text="员工每年享有5天带薪年假，满10年10天，满20年15天。",
            metadata={"file": "年假制度.pdf", "page": 1}
        ),
        Document(
            text="迟到早退超过30分钟按旷工半天处理。每月有3次迟到豁免机会。",
            metadata={"file": "考勤制度.pdf", "page": 2}
        ),
        Document(
            text="入职即缴纳五险一金，公积金个人12%公司12%。",
            metadata={"file": "福利制度.pdf", "page": 1}
        ),
        Document(
            text="年终奖根据个人绩效评定：A级6个月，B级3个月，C级1个月。",
            metadata={"file": "薪酬制度.pdf", "page": 1}
        ),
    ]
    print(f"    准备了 {len(documents)} 个文档")

    # 第 2 步：构建 Index
    print("\n  --- 第 2 步：构建 Index ---")
    index = VectorStoreIndex.from_documents(documents)
    print("    ✓ Index 构建完成")

    # 第 3 步：创建带 Postprocessor 的 QueryEngine
    print("\n  --- 第 3 步：创建 QueryEngine（带相似度过滤）---")
    query_engine = index.as_query_engine(
        similarity_top_k=3,                    # 检索前 3 个节点
        postprocessors=[                       # 添加后处理器
            SimilarityPostprocessor(           # 过滤低分节点
                similarity_cutoff=0.01
            )
        ],
        streaming=False                        # 非流式输出
    )
    print("    ✓ QueryEngine 创建完成")

    # 第 4 步：执行查询
    print("\n  --- 第 4 步：执行查询 ---")
    questions = [
        "年假几天？",
        "迟到怎么处罚？",
        "年终奖怎么算？",
    ]

    for question in questions:
        print(f"\n    问: {question}")
        response = query_engine.query(question)
        print(f"    答: {response.response}")
        print(f"    参考了 {len(response.source_nodes)} 个文档:")
        for src in response.source_nodes:
            print(f"      - {src.node.text[:40]}...")


# ============================================================================
# 第 8 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 7 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. QueryEngine = Retriever + Postprocessor + LLM 的封装                │
│  2. index.as_query_engine() 一行创建查询引擎                             │
│  3. query_engine.query() 返回 Response 对象（回答 + 源节点）             │
│  4. 可以自定义 Prompt 控制回答风格                                      │
│  5. 支持多种 response_mode（compact/refine）                            │
│  6. 支持流式输出（streaming=True）                                      │
│  7. 工作流：Documents → Index → QueryEngine → Response                  │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 创建 QueryEngine                                                     │
│  query_engine = index.as_query_engine(                                   │
│      similarity_top_k=3,                                                 │
│      response_mode="compact",                                            │
│      streaming=False,                                                    │
│  )                                                                      │
│                                                                         │
│  # 执行查询                                                             │
│  response = query_engine.query("用户的问题")                             │
│  print(response.response)        # LLM 生成的回答                        │
│  print(response.source_nodes)    # 参考的源节点                          │
│                                                                         │
│  # 自定义 Prompt                                                        │
│  from llama_index.core.prompts import PromptTemplate                    │
│  custom_prompt = "你是一个助手。请回答：{query_str}\\n信息：{context_str}"│
│  query_engine = index.as_query_engine(                                   │
│      text_qa_template=PromptTemplate(custom_prompt)                      │
│  )                                                                      │
│                                                                         │
│  【下一课预告】                                                           │
│  第 8 课：Ingestion Pipeline / 摄取管线 — 一键完成整个数据流水线         │
│  类比 Java：Spring Batch 作业                                           │
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
║   第 7 课：Query Engine / 查询引擎 — 生成最终回答         ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. as_query_engine() 一行创建查询引擎                  ║
║   2. Response 对象详解                                   ║
║   3. 自定义 Prompt — 控制回答风格                        ║
║   4. QueryEngine 参数详解                                ║
║   5. 不同 response_mode 对比                             ║
║   6. 流式输出（Streaming）                               ║
║   7. 完整示例：从文档到回答                              ║
║   8. 总结                                               ║
║                                                          ║
║   前置知识：第 1-6 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：as_query_engine() ━━━")
    demo_as_query_engine()

    print("\n━━━ 第 2 节：Response 对象 ━━━")
    demo_response_object()

    print("\n━━━ 第 3 节：自定义 Prompt ━━━")
    demo_custom_prompt()

    print("\n━━━ 第 4 节：QueryEngine 参数 ━━━")
    demo_query_engine_params()

    print("\n━━━ 第 5 节：response_mode 对比 ━━━")
    demo_response_modes()

    print("\n━━━ 第 6 节：流式输出 ━━━")
    demo_streaming()

    print("\n━━━ 第 7 节：完整示例 ━━━")
    demo_complete_example()

    print("\n🎉 第 7 课完成！")
    print("   建议下一步：阅读 week03/code/p29-query-engine.ipynb")
