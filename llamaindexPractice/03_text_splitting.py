"""
03_text_splitting.py — 文本分块策略

这一课深入学习 LlamaIndex 的文本分块（Chunking）：
1. 为什么需要分块？
2. 三种分块策略：Recursive, Token, Sentence
3. 分块参数调优
4. 分块质量评估

【Java 程序员速查】
  文本分块 = 把大文档切成小块，方便后续处理和检索
  类比 Java 中的字符串切割：
    String[] chunks = text.split("\\n");
  但 LlamaIndex 的分块更智能：考虑语义完整性、token 计数、重叠等
"""

import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import (
    Settings,
    Document,
)
from llama_index.core.node_parser import (
    SentenceSplitter,       # 句子分块器（最常用）
    TokenTextSplitter,     # Token 分块器（最精确）
    HierarchicalNodeParser,  # 层次分块器（适合长文档）
)
from llama_index.core.schema import TextNode  # 文本节点对象

# 配置 LLM（虽然这课主要关注分块，但需要配置完整）
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
print("  文本分块策略 — Text Splitting")
print("=" * 60)


# ============================================================
# 1. 为什么需要分块？
# ============================================================

def explain_why_chunking():
    """
    分块的必要性

    问题：
      1. LLM 有上下文窗口限制（GPT-3.5: 4096 tokens, GPT-4: 128K tokens）
      2. Embedding 模型对长文本效果差（超过 8192 token 的嵌入质量骤降）
      3. 小块检索更精准（检索整个文档 = 大海捞针）

    类比 Java：
      // 想象你要搜索一本 1000 页的书
      // 方法 A: 把整本书加载到内存 → 内存溢出
      // 方法 B: 按章节切分 → 只加载相关章节 → 高效

    分块策略选择：
      - 短文档（< 1000 字）: 不分块，整体处理
      - 中等文档（1K-10K 字）: 按段落或句子分块
      - 长文档（> 10K 字）: 层次化分块（先按章节，再按段落）
    """
    print("\n>>> 为什么需要分块？")
    print("""
    1. LLM 上下文窗口有限（GPT-3.5: 4096 tokens）
    2. Embedding 对长文本效果差
    3. 小块检索更精准

    类比：搜索一本书，是按章节查还是翻整本书？
    """)


# ============================================================
# 2. SentenceSplitter — 按句子分块（最常用）
# ============================================================

def demo_sentence_splitter():
    """
    SentenceSplitter 是最常用的分块器

    工作原理：
    1. 按句子边界分割（遇到 。！？.!? 就分割）
    2. 合并短句子直到达到 chunk_size
    3. 如果单个句子超过 chunk_size，按单词继续分割

    关键参数：
      chunk_size: 每个块的最大 token 数（默认 1024）
      chunk_overlap: 相邻块的重叠 token 数（默认 200）
      separator: 分割符（默认 " " 空格）
      paragraph_separator: 段落分割符（默认 "\n\n"）

    类比 Java:
      SentenceSplitter splitter = new SentenceSplitter(512, 50);
      List<TextSplit> splits = splitter.split(text);
    """
    print("\n>>> 使用 SentenceSplitter 分块")

    # 模拟一段较长的文本
    sample_text = """
    LlamaIndex 是一个强大的框架，用于构建基于大型语言模型（LLM）的应用。
    它提供了文档加载、文本分块、嵌入、索引、检索和查询等完整功能。

    在 RAG（检索增强生成）架构中，文本分块是关键的第一步。
    好的分块策略可以显著提升检索质量和回答准确性。

    分块时需要考虑的因素：
    1. 块大小：太小会丢失上下文，太大会引入噪声
    2. 重叠：适当的重叠可以保持语义连续性
    3. 分割边界：按句子、段落或语义单元分割

    LlamaIndex 提供了多种分块策略：
    - SentenceSplitter: 按句子分割，适合一般文本
    - TokenTextSplitter: 按 token 分割，最精确
    - HierarchicalNodeParser: 层次化分块，适合长文档
    """

    # --- 【Python 语法】SentenceSplitter ---
    # 创建分块器实例
    splitter = SentenceSplitter(
        chunk_size=200,    # 每个块最多 200 个 token
        chunk_overlap=20,  # 相邻块重叠 20 个 token
    )

    # --- 【Python 语法】.split_text() ---
    # 对文本进行分块，返回字符串列表
    # 类比 Java: splitter.splitText(text)
    chunks = splitter.split_text(sample_text)

    print(f"\n原文长度: {len(sample_text)} 字符")
    print(f"分块数量: {len(chunks)} 块")

    # --- 【Python 语法】enumerate() ---
    # 同时获取索引和值
    # 类比 Java: for (int i = 0; i < chunks.size(); i++)
    for i, chunk in enumerate(chunks):
        print(f"\n  块 {i+1} ({len(chunk)} 字符):")
        # 截断显示，避免输出太长
        preview = chunk if len(chunk) <= 100 else chunk[:100] + "..."
        print(f"    {preview}")

    # --- 【Python 语法】列表推导式 ---
    # 统计每个块的 token 数（近似为字符数/4）
    token_counts = [len(c) // 4 for c in chunks]
    print(f"\n各块 token 数（近似）: {token_counts}")
    print(f"平均 token 数: {sum(token_counts) / len(token_counts):.1f}")

    return chunks


# ============================================================
# 3. TokenTextSplitter — 按 Token 分块（最精确）
# ============================================================

def demo_token_splitter():
    """
    TokenTextSplitter 按 token 数量精确分块

    什么是 token？
    - Token 是 LLM 处理文本的最小单位
    - 英文：1 token ≈ 0.75 单词
    - 中文：1 token ≈ 1-2 汉字
    - GPT-4: 1 token = 4 个字符（近似）

    TokenTextSplitter 的优势：
    1. 精确控制每个块的 token 数
    2. 不会在 token 边界截断（避免损坏 embedding）
    3. 适合对 token 数敏感的场景

    类比 Java:
      TokenTextSplitter splitter = new TokenTextSplitter(500, 50);
      // 它会使用 tiktoken 库精确计算 token 数
    """
    print("\n>>> 使用 TokenTextSplitter 分块")

    sample_text = """
    机器学习是一种人工智能技术，它让计算机能够从数据中学习规律，
    而不需要显式编程。深度学习是机器学习的子领域，使用多层神经网络
    来学习数据的层次化表示。

    自然语言处理（NLP）是机器学习的另一个重要应用领域。
    NLP 让计算机能够理解和生成人类语言。
    大语言模型（LLM）如 GPT、Claude、Gemini 等都是 NLP 技术的巅峰之作。
    """

    # --- 【Python 语法】TokenTextSplitter ---
    splitter = TokenTextSplitter(
        chunk_size=100,    # 每个块最多 100 个 token
        chunk_overlap=10,  # 重叠 10 个 token
    )

    # --- 【Python 语法】.split_text() ---
    chunks = splitter.split_text(sample_text)

    print(f"\n原文长度: {len(sample_text)} 字符")
    print(f"分块数量: {len(chunks)} 块")

    for i, chunk in enumerate(chunks):
        # --- 【Python 语法】len() ---
        # 获取字符串长度（字符数）
        print(f"  块 {i+1}: {len(chunk)} 字符")
        print(f"    {chunk[:80]}...")

    return chunks


# ============================================================
# 4. HierarchicalNodeParser — 层次化分块
# ============================================================

def demo_hierarchical_splitter():
    """
    HierarchicalNodeParser 适合长文档的分块

    工作原理：
    1. 首先按层级标题分割（# 一级标题, ## 二级标题, ### 三级标题）
    2. 在每个层级内，再按 chunk_size 分割
    3. 子节点继承父节点的元数据

    适用场景：
    - 书籍、报告、论文等长文档
    - 需要保持文档结构信息的场景

    类比 Java:
      // 先按章节分割
      List<Chapter> chapters = parseByHeadings(text);
      // 再按段落分割
      for (Chapter chapter : chapters) {
          List<Paragraph> paragraphs = parseByParagraphs(chapter.getContent());
      }
    """
    print("\n>>> 使用层次化分块器")

    # 模拟一个有层级的长文档
    sample_text = """
    # 第一章：引言

    LlamaIndex 是一个用于构建 LLM 应用的框架。
    它提供了从文档加载到查询响应的完整流水线。

    ## 1.1 背景

    大语言模型的出现改变了人机交互的方式。
    人们可以用自然语言与 AI 对话，完成各种任务。

    ### 1.1.1 应用场景

    - 客服机器人
    - 文档问答
    - 代码生成
    - 数据分析

    ## 1.2 核心技术

    LlamaIndex 的核心技术包括：

    ### 2.1 文档加载

    支持多种文件格式的自动解析。

    ### 2.2 文本分块

    将长文档切分为小块，便于检索和处理。

    ### 2.3 向量嵌入

    将文本转换为高维向量，用于相似度搜索。

    ### 2.4 索引构建

    建立高效的向量索引，加速检索过程。

    ### 2.5 查询引擎

    提供统一的查询接口，支持多种检索策略。
    """

    # --- 【Python 语法】HierarchicalNodeParser ---
    # 创建层次化分块器
    parser = HierarchicalNodeParser.from_default_config(
        chunk_size=200,        # 每个块的最大 token 数
        chunk_overlap=20,      # 重叠 token 数
        separator="\n\n",      # 段落分隔符
    )

    # --- 【Python 语法】.get_nodes_from_documents() ---
    # 从文档列表获取分块节点
    # 类比 Java: parser.getNodesFromDocuments(documents)
    nodes = parser.get_nodes_from_documents([
        Document(text=sample_text)
    ])

    print(f"\n层次化分块结果:")
    print(f"  总节点数: {len(nodes)}")

    # --- 【Python 语法】TextNode 对象 ---
    # node.id_ — 节点 ID
    # node.text — 节点文本
    # node.metadata — 元数据（包含层级信息）
    # node.parent_node — 父节点
    # node.children_nodes — 子节点列表
    for i, node in enumerate(nodes):
        level = node.metadata.get("level", "?")
        title = node.metadata.get("title", "")
        preview = node.text[:60] + "..." if len(node.text) > 60 else node.text
        print(f"  节点 {i+1} [层级:{level}] {title}")
        print(f"    {preview}")


# ============================================================
# 5. 分块策略对比
# ============================================================

def compare_strategies():
    """
    对比三种分块策略的特点

    选择建议：
    - 一般文本 → SentenceSplitter（简单好用）
    - 精确控制 → TokenTextSplitter（最准确）
    - 长文档/书籍 → HierarchicalNodeParser（保持结构）
    """
    print("\n" + "=" * 60)
    print("  分块策略对比")
    print("=" * 60)
    print("""
    ┌─────────────────────┬──────────────┬────────────┬──────────────┐
    │      策略           │   适用场景    │   优点      │    缺点       │
    ├─────────────────────┼──────────────┼────────────┼──────────────┤
    │ SentenceSplitter    │ 一般文本      │ 简单快速    │ 可能跨句子截断 │
    │ TokenTextSplitter   │ 精确控制      │ 最准确      │ 需要 tiktoken  │
    │ HierarchicalParser  │ 长文档/书籍   │ 保持结构    │ 配置复杂      │
    └─────────────────────┴──────────────┴────────────┴──────────────┘

    推荐配置：
    - chunk_size: 256-512（一般场景）
    - chunk_overlap: 20-50（保持语义连续性）
    - 中文文本可适当增大 chunk_size
    """)


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数：依次演示各种分块策略
    """
    explain_why_chunking()
    demo_sentence_splitter()
    demo_token_splitter()
    demo_hierarchical_splitter()
    compare_strategies()

    print("\n" + "=" * 60)
    print("  OK 文本分块策略完成！")
    print("=" * 60)
    print("""
下一步：
  - 根据你的文档类型选择合适的分块策略
  - 调整 chunk_size 和 chunk_overlap 参数
  - 观察分块结果对检索质量的影响
    """)


if __name__ == "__main__":
    main()
