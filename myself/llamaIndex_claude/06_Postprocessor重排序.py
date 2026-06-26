"""
===============================================================================
 第 6 课：Postprocessor / 重排序 — 提升检索质量
===============================================================================

【这一课学什么？】
  第 5 课我们学会了用 Retriever 检索相关文档。但向量相似度搜索有个问题：
  它找到的"最相似"结果不一定是最有用的。
  Postprocessor 就是在检索之后、回答之前，对结果进行二次筛选和排序。

【类比 Java】
  Postprocessor ≈ 二级排序 / 过滤器
  - 第 1 轮：向量相似度搜索（召回 Top-K，速度快但粗糙）
  - 第 2 轮：Postprocessor 精排（过滤低分、重排序、关键词匹配）

  类比 SQL：
    -- 第 1 轮：向量搜索（召回）
    SELECT * FROM nodes ORDER BY cosine_similarity(vec, ?) LIMIT 20

    -- 第 2 轮：Postprocessor（精排）
    SELECT * FROM (...) WHERE similarity > 0.5 ORDER BY updated_at DESC

【核心概念】
  Postprocessor 的工作流程：
    Retriever → List[NodeWithScore] → Postprocessor → 过滤后的 List[NodeWithScore]

  常用 Postprocessor：
    - SimilarityPostprocessor  → 过滤低于阈值的节点
    - KeywordNodePostprocessor → 只保留包含关键词的节点
    - LLMRerank                → 用 LLM 重新排序（最准但最慢）
    - SentenceTransformerRerank → 用交叉编码器重排（性价比高）

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/06_Postprocessor重排序.py

【前置知识】
  - 第 1 课：Node / 文本切片
  - 第 2 课：Settings / 全局配置
  - 第 5 课：Index + Retriever
"""

import os
from llama_index.core import Settings


# ============================================================================
# 第 1 节：为什么需要 Postprocessor？
# ============================================================================

def demo_why_postprocessor():
    """
    演示向量相似度搜索的局限性

    向量搜索的问题是：它只看语义相似度，不看其他因素。
    比如：
    - 相似度很低的节点也可能被返回
    - 过时的节点可能排在前面
    - 不包含关键词的相关节点可能被漏掉
    """
    print("=" * 60)
    print("【为什么需要 Postprocessor？】")
    print("=" * 60)

    print("""
  假设我们有以下文档：
    Doc A: "2024年公司年会将于12月25日举行"
    Doc B: "员工每年享有5天带薪年假"
    Doc C: "2023年公司年会回顾"
    Doc D: "猫是哺乳动物"

  查询: "公司年会什么时候？"

  向量相似度搜索可能返回：
    1. Doc C (2023年会回顾) — 语义上很像，但是过时信息！
    2. Doc A (2024年会通知) — 正确答案
    3. Doc B (年假制度)     — 完全不相关，但文本长度相似
    4. Doc D (猫)           — 完全不相关

  Postprocessor 可以做：
    ✓ SimilarityPostprocessor: 过滤掉相似度 < 0.5 的节点
    ✓ FixedRecencyPostprocessor: 把最新的节点排在前面
    ✓ KeywordNodePostprocessor: 只保留包含"年会"的节点

  经过 Postprocessor 后：
    1. Doc A (2024年会) — 正确！
    2. Doc C (2023年会) — 被时效性过滤降序
    3. Doc B, Doc D   — 被相似度过滤移除
    """)


# ============================================================================
# 第 2 节：SimilarityPostprocessor — 相似度过滤
# ============================================================================

def demo_similarity_postprocessor():
    """
    SimilarityPostprocessor 是最简单的 Postprocessor

    它只保留相似度高于某个阈值的节点。
    类比 Java：
      results.stream()
          .filter(r -> r.getScore() > 0.5)
          .collect(toList());
    """
    print("=" * 60)
    print("【SimilarityPostprocessor — 相似度过滤】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.postprocessor import SimilarityPostprocessor

    # 准备文档
    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5-15天带薪年假。"),
        Document(text="猫喜欢吃鱼。"),  # 不相关的文档
        Document(text="五险一金按基数12%缴纳。"),
    ]

    # 创建 Index
    index = VectorStoreIndex.from_documents(documents)

    # 创建不带过滤的检索器
    print("\n  --- 不带过滤的检索结果 ---")
    retriever_raw = index.as_retriever(similarity_top_k=3)
    nodes_raw = retriever_raw.retrieve("工作时间")
    print(f"    查询: '工作时间'")
    print(f"    返回 {len(nodes_raw)} 个结果:")
    for i, node_score in enumerate(nodes_raw):
        print(f"      {i + 1}. [{node_score.score:.4f}] {node_score.node.text}")

    # 创建带相似度过滤的检索器
    print("\n  --- 带相似度过滤的检索结果 ---")
    # SimilarityPostprocessor 参数：
    #   similarity_cutoff → 相似度阈值（0-1之间）
    #   similarity_top_k  → 最多返回多少个
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever

    # 方式 1：直接在 as_retriever 中使用 postprocessors
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.1)

    # 注意：as_retriever 不直接支持 postprocessors 参数
    # 需要使用 RetrieverQueryEngine 来组合
    retriever = index.as_retriever(similarity_top_k=3)
    nodes_filtered = postprocessor.postprocess_nodes(
        [n for n in nodes_raw],  # 从原始检索结果中过滤
        query_str="工作时间",
        query_bundle=None
    )

    print(f"    阈值: 0.1")
    print(f"    过滤后返回 {len(nodes_filtered)} 个结果:")
    for i, node_score in enumerate(nodes_filtered):
        print(f"      {i + 1}. [{node_score.score:.4f}] {node_score.node.text}")


# ============================================================================
# 第 3 节：KeywordNodePostprocessor — 关键词过滤
# ============================================================================

def demo_keyword_postprocessor():
    """
    KeywordNodePostprocessor 只保留包含指定关键词的节点

    类比 Java：
      results.stream()
          .filter(r -> r.getText().contains("考勤"))
          .collect(toList());
    """
    print("=" * 60)
    print("【KeywordNodePostprocessor — 关键词过滤】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.postprocessor import KeywordNodePostprocessor

    # 准备文档
    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
        Document(text="考勤制度：迟到扣款100元。"),
        Document(text="猫是哺乳动物。"),
    ]

    # 创建 Index
    index = VectorStoreIndex.from_documents(documents)

    # 创建关键词过滤器
    print("\n  --- 只保留包含'考勤'或'工作'的节点 ---")
    keyword_processor = KeywordNodePostprocessor(
        required_keywords=["考勤", "工作"],  # 必须包含这些词
        exclude_keywords=["猫"]              # 不能包含这些词
    )

    # 检索
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("工作制度")

    # 应用关键词过滤
    filtered = keyword_processor.postprocess_nodes(
        list(nodes),
        query_str="工作制度",
        query_bundle=None
    )

    print(f"    原始结果: {len(nodes)} 个")
    print(f"    过滤后: {len(filtered)} 个")
    for i, node_score in enumerate(filtered):
        print(f"      {i + 1}. {node_score.node.text}")

    print("""
  💡 适用场景：
    - 需要确保结果包含特定术语
    - 排除不相关的内容
    - 与向量搜索结合，提高精确度
    """)


# ============================================================================
# 第 4 节：LLMRerank — 用 LLM 重排序
# ============================================================================

def demo_llm_rerank():
    """
    LLMRerank 是最强大的 Postprocessor

    它让 LLM 亲自判断每个节点与查询的相关性，然后重新排序。
    效果最好，但速度最慢、成本最高。

    类比 Java：
      // 用一个智能的评分器来给每个结果打分
      for (Result r : results) {
          r.score = llm.evaluate(relevance, query, r.text);
      }
      results.sort(Comparator.comparingDouble(Result::getScore).reversed());
    """
    print("=" * 60)
    print("【LLMRerank — 用 LLM 重排序】")
    print("=" * 60)

    from llama_index.core.postprocessor import LLMRerank

    # 检查 LLM 是否配置
    if Settings.llm is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            from llama_index.llms.dashscope import DashScope
            Settings.llm = DashScope(
                model_name="qwen-plus",
                api_key=api_key,
                temperature=0.1,
            )
            print(f"    ✓ 已配置 LLM: {Settings.llm.model_name}")
        else:
            print("    ⚠️  缺少 DASHSCOPE_API_KEY，跳过 LLMRerank 演示\n")
            return

    print("\n  --- LLMRerank 的工作原理 ---")
    print("""
    1. 检索器先召回 Top-K 个节点（比如 10 个）
    2. LLMRerank 把查询 + 每个节点文本发给 LLM
    3. LLM 判断每个节点与查询的相关性（0-1 分）
    4. 按 LLM 打分重新排序
    5. 返回重新排序后的结果

    示例：
      查询: "年假几天？"

      原始检索结果（按向量相似度排序）:
        1. "员工每年享有5天带薪年假"          → 0.85
        2. "工作满10年年假增至10天"           → 0.72
        3. "猫是哺乳动物"                     → 0.15  (低分)
        4. "五险一金按基数缴纳"               → 0.10  (低分)

      LLMRerank 重新打分后:
        1. "员工每年享有5天带薪年假"          → 0.95 (最相关)
        2. "工作满10年年假增至10天"           → 0.88 (相关)
        3. "猫是哺乳动物"                     → 0.02 (不相关，降序)
        4. "五险一金按基数缴纳"               → 0.01 (不相关，降序)

    可以看到：LLM 能理解语义，更准确地判断相关性！
    """)


# ============================================================================
# 第 5 节：SentenceTransformerRerank — 性价比之选
# ============================================================================

def demo_sentence_transformer_rerank():
    """
    SentenceTransformerRerank 使用交叉编码器（Cross-Encoder）重排序

    特点：
    - 比 LLMRerank 快很多
    - 比纯向量搜索准
    - 不需要 LLM API 调用（本地运行）
    - 需要安装 sentence-transformers 库

    类比 Java：
      // 用一个轻量级的评分模型，比向量相似度准，但比 LLM 快
      for (Result r : results) {
          r.score = crossEncoder.score(query, r.text);
      }
    """
    print("=" * 60)
    print("【SentenceTransformerRerank — 性价比之选】")
    print("=" * 60)

    print("""
  安装：
    pip install sentence-transformers
    pip install llama-index-postprocessor-sentence-rerank

  使用：
    from llama_index.postprocessor.sentence_rerank import SentenceRerankPostprocessor

    reranker = SentenceRerankPostprocessor(
        model="bge-reranker-v2-m3",  # 中文效果好
        top_n=3                       # 只保留前 3 个
    )

  特点：
    - 速度：比 LLMRerank 快 10-100 倍
    - 精度：比纯向量搜索高 10-20%
    - 成本：完全免费（本地运行）
    - 离线：不需要联网

  适用场景：
    - 生产环境首选（速度+精度的平衡）
    - 不想花 LLM API 费用
    - 需要离线运行
    """)


# ============================================================================
# 第 6 节：组合多个 Postprocessor
# ============================================================================

def demo_combined_postprocessors():
    """
    可以组合多个 Postprocessor，形成多级过滤

    类比 Java：
      results.stream()
          .filter(r -> r.getScore() > 0.5)        // 第一级：相似度过滤
          .filter(r -> r.getText().contains("考勤")) // 第二级：关键词过滤
          .sorted(Comparator.comparingDouble(Result::getScore).reversed()) // 第三级：重排序
          .limit(3)                                // 第四级：限制数量
          .collect(toList());
    """
    print("=" * 60)
    print("【组合多个 Postprocessor】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
    from llama_index.core.query_engine import RetrieverQueryEngine

    # 准备文档
    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
        Document(text="考勤制度：迟到扣款100元。"),
        Document(text="猫是哺乳动物。"),
    ]

    # 创建 Index
    index = VectorStoreIndex.from_documents(documents)

    # 创建多个 Postprocessor
    print("\n  --- 组合过滤链 ---")
    print("  第 1 级：SimilarityPostprocessor (过滤低分)")
    print("  第 2 级：KeywordNodePostprocessor (保留含'工作'的)")

    postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=0.05),
        KeywordNodePostprocessor(required_keywords=["工作"]),
    ]

    # 创建带 postprocessors 的 QueryEngine
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        postprocessors=postprocessors
    )

    # 执行查询
    response = query_engine.query("工作制度")
    print(f"\n  查询: '工作制度'")
    print(f"  回答: {response.response}")
    print(f"  使用了 {len(response.source_nodes)} 个源节点:")
    for i, src in enumerate(response.source_nodes):
        print(f"    {i + 1}. {src.node.text}")


# ============================================================================
# 第 7 节：Postprocessor 对比总结
# ============================================================================

def demo_postprocessor_summary():
    """
    各种 Postprocessor 的对比
    """
    print("=" * 60)
    print("【Postprocessor 对比表】")
    print("=" * 60)

    print("""
  ┌──────────────────────────┬──────┬──────┬───────┬──────────────────┐
  │ Postprocessor            │ 速度 │ 精度 │ 成本  │ 适用场景          │
  ├──────────────────────────┼──────┼──────┼───────┼──────────────────┤
  │ SimilarityPostprocessor  │ ★★★★★│  ★★★ │  免费  │ 过滤低分结果      │
  │ KeywordNodePostprocessor │ ★★★★★│  ★★★ │  免费  │ 关键词精确匹配    │
  │ FixedRecencyPostprocessor│ ★★★★ │  ★★★ │  免费  │ 时效性内容        │
  │ SentenceTransformerRerank│ ★★★  │  ★★★★│  免费  │ 生产环境首选      │
  │ LLMRerank                │ ★    │  ★★★★★│ 有成本│ 最高精度要求      │
  └──────────────────────────┴──────┴──────┴───────┴──────────────────┘

  推荐组合（生产环境）：
    SimilarityPostprocessor + KeywordNodePostprocessor + SentenceTransformerRerank

  推荐组合（开发/原型）：
    只用 VectorStoreIndex + as_retriever()，不加 Postprocessor
    """)


# ============================================================================
# 第 8 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 6 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. Postprocessor 在检索后对结果进行二次筛选和排序                       │
│  2. SimilarityPostprocessor 过滤低相似度节点                             │
│  3. KeywordNodePostprocessor 保留包含关键词的节点                        │
│  4. LLMRerank 用 LLM 重新打分排序（最准但最慢）                          │
│  5. SentenceTransformerRerank 性价比最高                                │
│  6. 可以组合多个 Postprocessor 形成过滤链                                │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 相似度过滤                                                            │
│  from llama_index.core.postprocessor import SimilarityPostprocessor     │
│  processor = SimilarityPostprocessor(similarity_cutoff=0.5)             │
│                                                                         │
│  # 关键词过滤                                                           │
│  from llama_index.core.postprocessor import KeywordNodePostprocessor    │
│  processor = KeywordNodePostprocessor(                                   │
│      required_keywords=["考勤", "工作"],                                 │
│      exclude_keywords=["猫"]                                             │
│  )                                                                      │
│                                                                         │
│  # 组合使用                                                             │
│  query_engine = index.as_query_engine(                                   │
│      similarity_top_k=5,                                                 │
│      postprocessors=[processor1, processor2]                             │
│  )                                                                      │
│                                                                         │
│  【下一课预告】                                                           │
│  第 7 课：Query Engine / 查询引擎 — 生成最终回答                         │
│  类比 Java：Service 层组装响应                                           │
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
║   第 6 课：Postprocessor / 重排序 — 提升检索质量          ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. 为什么需要 Postprocessor？                           ║
║   2. SimilarityPostprocessor — 相似度过滤                ║
║   3. KeywordNodePostprocessor — 关键词过滤               ║
║   4. LLMRerank — 用 LLM 重排序                          ║
║   5. SentenceTransformerRerank — 性价比之选              ║
║   6. 组合多个 Postprocessor                              ║
║   7. Postprocessor 对比总结                              ║
║   8. 总结                                               ║
║                                                          ║
║   前置知识：第 1-5 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：为什么需要 Postprocessor？ ━━━")
    demo_why_postprocessor()

    print("\n━━━ 第 2 节：SimilarityPostprocessor ━━━")
    demo_similarity_postprocessor()

    print("\n━━━ 第 3 节：KeywordNodePostprocessor ━━━")
    demo_keyword_postprocessor()

    print("\n━━━ 第 4 节：LLMRerank ━━━")
    demo_llm_rerank()

    print("\n━━━ 第 5 节：SentenceTransformerRerank ━━━")
    demo_sentence_transformer_rerank()

    print("\n━━━ 第 6 节：组合 Postprocessor ━━━")
    demo_combined_postprocessors()

    print("\n━━━ 第 7 节：Postprocessor 对比 ━━━")
    demo_postprocessor_summary()

    print("\n🎉 第 6 课完成！")
    print("   建议下一步：阅读 week03/code/p28-postprocessor.ipynb")
