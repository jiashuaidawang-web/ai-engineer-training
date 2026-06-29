"""
06_query_engine.py — 查询引擎

这一课深入学习 LlamaIndex 的查询引擎：
1. SimpleQueryEngine — 最简单的查询
2. RetrieverQueryEngine — 检索 + 生成
3. ChatEngine — 对话式查询
4. 查询引擎的定制与调优

【Java 程序员速查】
  查询引擎 = 用户提问 → 系统回答 的完整管道
  类比 Java:
    // Spring MVC 的请求处理链路
    Controller → Service → Repository → Database → Response
    // LlamaIndex 的查询链路
    QueryEngine → Retriever → VectorStore → LLM → Response
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
from llama_index.core.prompts import PromptTemplate  # 提示模板
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,  # 响应合成器
    CompactAndRefine,          # 紧凑精炼策略
    Refine,                    # 精炼策略
    TreeSummarize,             # 树形摘要策略
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
print("  查询引擎 — Query Engine")
print("=" * 60)


# ============================================================
# 1. SimpleQueryEngine — 最简单的方式
# ============================================================

def demo_simple_query_engine():
    """
    SimpleQueryEngine 是最基础的查询引擎

    工作流程：
    1. 接收用户问题
    2. 从索引中检索相关文档
    3. 将文档 + 问题 发送给 LLM
    4. LLM 生成回答

    类比 Java:
      // 最简单的方式：直接调用索引的查询方法
      QueryEngine engine = index.asQueryEngine();
      QueryResponse response = engine.query("你的问题");
      System.out.println(response.getResponse());
    """
    print("\n>>> 使用 SimpleQueryEngine")

    # 创建测试文档
    documents = [
        Document(text="LlamaIndex 是一个用于构建 LLM 应用的框架。"
                      "它提供文档加载、文本分块、嵌入、索引和查询功能。"),
        Document(text="RAG（检索增强生成）结合了信息检索和文本生成的优势。"
                      "先用向量数据库检索相关文档，再将文档作为上下文提供给 LLM。"),
        Document(text="向量数据库（如 Milvus、ChromaDB）专门用于存储和检索高维向量。"
                      "它们支持高效的相似度搜索，是 RAG 系统的核心组件。"),
    ]

    # 构建索引
    index = VectorStoreIndex.from_documents(documents)

    # --- 【Python 语法】.as_query_engine() ---
    # 将索引转换为查询引擎
    # 这是最常用的方式，一步完成所有配置
    # 类比 Java: index.asQueryEngine();
    query_engine = index.as_query_engine()

    # --- 【Python 语法】query_engine.query() ---
    # 发送查询，获取回答
    # 类比 Java: engine.query("你的问题");
    questions = [
        "什么是 LlamaIndex？",
        "RAG 的工作原理是什么？",
        "向量数据库在 RAG 中扮演什么角色？",
    ]

    for question in questions:
        print(f"\n  问题: {question}")
        print("-" * 40)

        # --- 【Python 语法】response.response ---
        # QueryResponse 对象的 response 属性包含 LLM 的回答
        # 类比 Java: response.getResponse()
        response = query_engine.query(question)
        print(f"  回答: {response.response}")

        # --- 【Python 语法】response.source_nodes ---
        # 检索到的相关节点（用于追溯和调试）
        # 类比 Java: response.getSourceNodes()
        if hasattr(response, 'source_nodes'):
            print(f"  参考了 {len(response.source_nodes)} 个文档块")
            for i, node in enumerate(response.source_nodes):
                print(f"    节点 {i+1} 相似度: {node.score:.4f}")


# ============================================================
# 2. RetrieverQueryEngine — 自定义检索
# ============================================================

def demo_retriever_query_engine():
    """
    RetrieverQueryEngine 允许你自定义检索逻辑

    工作流程：
    1. 创建检索器（Retriever）
    2. 创建响应合成器（Response Synthesizer）
    3. 组合成查询引擎

    类比 Java:
      // 自定义检索器
      Retriever retriever = index.asRetriever();
      retriever.setSimilarityTopK(5);

      // 自定义响应合成器
      ResponseSynthesizer synthesizer = new CompactAndRefine();

      // 组合
      RetrieverQueryEngine engine = RetrieverQueryEngine.fromDefaults(
          retriever, synthesizer
      );
    """
    print("\n>>> 使用 RetrieverQueryEngine")

    documents = [
        Document(text="Python 是一种广泛使用的高级编程语言，以代码简洁著称。"),
        Document(text="Java 是一种强类型的面向对象编程语言，广泛应用于企业级开发。"),
        Document(text="JavaScript 是 Web 开发的主要语言，支持前端和后端（Node.js）。"),
        Document(text="Go 语言以其并发模型和高性能著称，适合构建分布式系统。"),
    ]

    # 构建索引
    index = VectorStoreIndex.from_documents(documents)

    # --- 【Python 语法】创建检索器 ---
    retriever = index.as_retriever(
        similarity_top_k=3,    # 返回最相似的 3 个节点
        similarity_cutoff=0.5, # 相似度低于 0.5 的不返回
    )

    # --- 【Python 语法】创建响应合成器 ---
    # CompactAndRefine: 先压缩文档，再逐步精炼回答
    # Refine: 对每个文档块生成部分回答，然后合并
    # TreeSummarize: 构建树形结构进行摘要
    from llama_index.core.response_synthesizers import CompactAndRefine
    synthesizer = CompactAndRefine(
        verbose=True,  # 打印详细日志
    )

    # --- 【Python 语法】RetrieverQueryEngine.from_defaults() ---
    # 从检索器和合成器创建查询引擎
    # 类比 Java: RetrieverQueryEngine.fromDefaults(retriever, synthesizer);
    from llama_index.core.query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine.from_defaults(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )

    # 查询
    response = query_engine.query("哪种编程语言最适合初学者？")
    print(f"\n  问题: 哪种编程语言最适合初学者？")
    print(f"  回答: {response.response}")


# ============================================================
# 3. ChatEngine — 对话式查询
# ============================================================

def demo_chat_engine():
    """
    ChatEngine 支持多轮对话

    与普通查询引擎的区别：
    - 普通查询：一问一答，无上下文
    - ChatEngine：记住对话历史，支持追问

    类比 Java:
      // 普通查询：每次独立
      response = engine.query("问题1");
      response = engine.query("问题2");

      // ChatEngine：带对话历史
      chatEngine = index.asChatEngine();
      chatEngine.chat("你好");
      chatEngine.chat("刚才说的能详细解释吗？");  // 能理解"刚才说的"指什么
    """
    print("\n>>> 使用 ChatEngine")

    documents = [
        Document(text="LlamaIndex 支持多种索引类型：VectorStoreIndex, SummaryIndex, KeywordTableIndex, TreeIndex。"),
        Document(text="VectorStoreIndex 是最常用的索引类型，基于向量相似度搜索。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # --- 【Python 语法】.as_chat_engine() ---
    # 创建聊天引擎
    # 参数：
    #   chat_mode: 聊天模式（"openai" 使用 LLM 的对话功能）
    #   memory: 对话记忆（记录历史消息）
    #   llm: 指定 LLM
    # 类比 Java: index.asChatEngine(ChatMode.OPENAI);
    chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)

    # --- 【Python 语法】chat_engine.chat() ---
    # 发送聊天消息，返回 ChatResponse
    # 类比 Java: chatEngine.chat("你的消息");
    conversations = [
        "介绍一下 LlamaIndex 支持的索引类型",
        "VectorStoreIndex 有什么特点？",
        "它和其他索引相比有什么优势？",  # 这里的"它"指的是 VectorStoreIndex
    ]

    for msg in conversations:
        print(f"\n  用户: {msg}")
        response = chat_engine.chat(msg)
        print(f"  助手: {response.response}")


# ============================================================
# 4. 自定义提示模板
# ============================================================

def demo_custom_prompts():
    """
    自定义提示模板 = 告诉 LLM 如何回答

    默认情况下，LlamaIndex 使用内置的提示模板。
    你可以自定义模板来控制回答的风格和格式。

    类比 Java:
      // 自定义 Prompt Template
      PromptTemplate template = new PromptTemplate(
          "请基于以下参考信息回答问题：\n{context}\n\n问题：{query}\n回答："
      );
    """
    print("\n>>> 使用自定义提示模板")

    documents = [
        Document(text="LlamaIndex 是一个用于构建 LLM 应用的框架。"),
        Document(text="它支持文档加载、文本分块、嵌入、索引和查询。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # --- 【Python 语法】PromptTemplate ---
    # 创建自定义提示模板
    # {context_str} 会被替换为检索到的文档内容
    # {query_str} 会被替换为用户的问题
    custom_prompt = PromptTemplate(
        "你是一个专业的技术助手。请基于以下参考信息回答问题。\n"
        "参考信息：{context_str}\n"
        "用户问题：{query_str}\n"
        "请给出简洁、准确的回答："
    )

    # --- 【Python 语法】.as_query_engine(prompt_template=...) ---
    # 将自定义提示模板应用到查询引擎
    query_engine = index.as_query_engine(
        text_qa_template=custom_prompt,  # 文本问答模板
    )

    response = query_engine.query("LlamaIndex 是什么？")
    print(f"\n  问题: LlamaIndex 是什么？")
    print(f"  回答: {response.response}")


# ============================================================
# 5. 查询引擎对比
# ============================================================

def compare_engines():
    """
    三种查询引擎对比

    选择建议：
    - SimpleQueryEngine: 快速原型，简单问答
    - RetrieverQueryEngine: 需要自定义检索逻辑
    - ChatEngine: 多轮对话场景
    """
    print("\n" + "=" * 60)
    print("  查询引擎对比")
    print("=" * 60)
    print("""
    ┌─────────────────────┬──────────────┬──────────────┬──────────────┐
    │      引擎           │   适用场景    │   优点        │    缺点       │
    ├─────────────────────┼──────────────┼──────────────┼──────────────┤
    │ SimpleQueryEngine   │ 简单问答      │ 最简单        │ 无对话历史    │
    │ RetrieverQueryEngine│ 自定义检索    │ 灵活可控      │ 配置复杂      │
    │ ChatEngine          │ 多轮对话      │ 支持追问      │ 消耗更多 token │
    └─────────────────────┴──────────────┴──────────────┴──────────────┘

    推荐：
    - 入门：SimpleQueryEngine
    - 进阶：RetrieverQueryEngine + 自定义提示
    - 对话：ChatEngine
    """)


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数：依次演示各种查询引擎
    """
    demo_simple_query_engine()
    demo_retriever_query_engine()
    demo_chat_engine()
    demo_custom_prompts()
    compare_engines()

    print("\n" + "=" * 60)
    print("  OK 查询引擎完成！")
    print("=" * 60)
    print("""
下一步：
  - 理解不同查询引擎的适用场景
  - 尝试自定义提示模板
  - 下一课：完整 RAG 管道
    """)


if __name__ == "__main__":
    main()
