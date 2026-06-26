"""
===============================================================================
 第 9 课：Chat Engine + Memory / 对话引擎与记忆
===============================================================================

【这一课学什么？】
  第 7 课的 QueryEngine 是一次性问答——问一个问题，得到一个回答。
  但真实的聊天场景需要"记忆"：
    用户: "年假几天？"
    AI: "5天"
    用户: "那满10年呢？"    ← 这里的"那"指的是年假！

  Chat Engine 让 LlamaIndex 支持多轮对话，Memory 让 AI 记住历史。

【类比 Java】
  Chat Engine ≈ HttpSession（会话状态管理）
  Memory ≈ 存储在 Session 中的对话历史
  ContextChatEngine ≈ 每次对话都带上相关知识库上下文

【核心概念】
  Chat Engine 的类型：
    - ConversableQueryEngineChatEngine → 基于 QueryEngine 的对话
    - ContextChatEngine → 每次对话都检索知识库
    - SimpleChatEngine → 最简单的对话（不调用知识库）

  Memory 的类型：
    - ChatMemoryBuffer → 简单的消息缓冲区
    - VectorMemory → 基于向量存储的记忆

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/09_ChatEngine对话引擎.py

【前置知识】
  第 1-8 课的所有知识
"""

import os
from llama_index.core import Settings


# ============================================================================
# 第 1 节：为什么需要 Chat Engine？
# ============================================================================

def demo_why_chat_engine():
    """
    演示一次性问答 vs 多轮对话的区别

    没有 Chat Engine（一次性问答）：
      问: 年假几天？  → 答: 5天
      问: 满10年呢？ → 答: 我不知道你在说什么（没有上下文！）

    有 Chat Engine（多轮对话）：
      问: 年假几天？  → 答: 5天
      问: 满10年呢？ → 答: 满10年是10天（知道"那"指年假！）
    """
    print("=" * 60)
    print("【为什么需要 Chat Engine？】")
    print("=" * 60)

    print("""
  场景对比：

  ❌ 没有 Chat Engine（每次都是独立问题）：
    用户: 年假几天？
    AI: 5天

    用户: 那满10年呢？     ← AI 不知道"那"指什么！
    AI: 我不明白你的问题。

  ✓ 有 Chat Engine（记住上下文）：
    用户: 年假几天？
    AI: 5天

    用户: 那满10年呢？     ← AI 知道"那"指年假！
    AI: 满10年是10天。

    用户: 病假呢？         ← AI 知道在问假期制度
    AI: 病假需要提供医院诊断证明。
    """)


# ============================================================================
# 第 2 节：ConversableQueryEngineChatEngine
# ============================================================================

def demo_conversable_chat_engine():
    """
    ConversableQueryEngineChatEngine 是基于 QueryEngine 的对话引擎

    它结合了：
    - QueryEngine 的知识库检索能力
    - Memory 的对话记忆能力
    """
    print("=" * 60)
    print("【ConversableQueryEngineChatEngine】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.chat_engine import ConversableQueryEngineChatEngine

    # 准备文档
    documents = [
        Document(text="员工每年享有5天带薪年假，满10年10天，满20年15天。"),
        Document(text="病假需提供二级以上医院诊断证明，按基本工资80%发放。"),
        Document(text="婚假15天，产假98天，陪产假15天。"),
    ]

    # 创建 Index
    index = VectorStoreIndex.from_documents(documents)

    # 创建 Chat Engine
    # 参数说明：
    #   index          → 关联的 Index
    #   memory         → 对话记忆（默认自动创建）
    #   token_limit    → 记忆保留的最大 token 数（防止无限增长）
    chat_engine = ConversableQueryEngineChatEngine(
        index=index,
        memory=None,              # None = 使用默认记忆
        token_limit=8192,         # 记忆最多保留 8192 个 token
    )

    # 模拟多轮对话
    print("\n  --- 多轮对话演示 ---")
    conversations = [
        "年假几天？",
        "那满10年呢？",
        "病假怎么请？",
        "婚假有几天？",
    ]

    for question in conversations:
        print(f"\n  用户: {question}")
        response = chat_engine.chat(question)
        print(f"  AI: {response.response}")

    print("\n  💡 注意：即使第二问'那满10年呢？'没有提到'年假'，")
    print("     Chat Engine 也能理解上下文！")


# ============================================================================
# 第 3 节：ContextChatEngine — 每次对话都检索知识库
# ============================================================================

def demo_context_chat_engine():
    """
    ContextChatEngine 在每次对话时都从知识库检索相关信息

    与 ConversableQueryEngineChatEngine 的区别：
    - Conversable: 先用 Index 检索，然后在对话中复用
    - Context: 每次对话都重新检索，更准确但更慢

    类比 Java：
      // Conversable: 检索一次，复用多次
      List<Node> nodes = retriever.retrieve(initialQuery);
      for (String msg : conversation) {
          answer = llm.generate(msg, context + nodes);
      }

      // Context: 每次对话都重新检索
      for (String msg : conversation) {
          List<Node> nodes = retriever.retrieve(msg);  // 每次都检索
          answer = llm.generate(msg, context + nodes);
      }
    """
    print("=" * 60)
    print("【ContextChatEngine — 每次对话都检索知识库】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.chat_engine import ContextChatEngine

    documents = [
        Document(text="公司实行每日八小时工作制。"),
        Document(text="员工每年享有5天带薪年假。"),
        Document(text="迟到超过30分钟按旷工处理。"),
    ]

    index = VectorStoreIndex.from_documents(documents)

    # 创建 ContextChatEngine
    # 参数说明：
    #   index           → 关联的 Index
    #   chat_mode       → 对话模式
    #     "condense_question" → 把当前问题和历史合并成一个查询
    #     "condense_plus_context" → 合并问题 + 检索上下文
    chat_engine = ContextChatEngine.from_defaults(
        index=index,
        chat_mode="condense_plus_context",
    )

    print("\n  --- 多轮对话演示 ---")
    conversations = [
        "工作时间是多久？",
        "年假呢？",
        "迟到怎么处罚？",
    ]

    for question in conversations:
        print(f"\n  用户: {question}")
        response = chat_engine.chat(question)
        print(f"  AI: {response.response}")


# ============================================================================
# 第 4 节：Memory — 对话记忆详解
# ============================================================================

def demo_memory():
    """
    Memory 是 Chat Engine 的核心组件

    它存储对话历史，让 AI 能记住之前说了什么。

    类比 Java：
      // HttpSession 存储对话状态
      HttpSession session = request.getSession();
      session.setAttribute("messages", messageHistory);
    """
    print("=" * 60)
    print("【Memory — 对话记忆详解】")
    print("=" * 60)

    from llama_index.core.memory import ChatMemoryBuffer

    # ChatMemoryBuffer 是最基本的记忆组件
    # 参数说明：
    #   token_limit   → 记忆保留的最大 token 数
    #   chat_store    → 可选：自定义消息存储
    memory = ChatMemoryBuffer(token_limit=1000)

    print("\n  --- 模拟对话记忆 ---")

    # 添加对话消息到记忆
    # 类比 Java: memory.addMessage(new HumanMessage("你好"));
    memory.put({"role": "user", "content": "年假几天？"})
    memory.put({"role": "assistant", "content": "5天"})

    memory.put({"role": "user", "content": "那满10年呢？"})
    memory.put({"role": "assistant", "content": "10天"})

    # 获取记忆中的所有消息
    messages = memory.get()
    print(f"  记忆中有 {len(messages)} 条消息:")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:30]
        print(f"    [{role}] {content}")

    print("""
  Memory 的生命周期：
    1. 用户发送消息 → memory.put(user_message)
    2. AI 生成回答 → memory.put(assistant_message)
    3. 下一轮对话 → memory.get() 获取所有历史消息
    4. 超过 token_limit → 删除最早的消息

  类比 Java:
    Deque<Message> history = new ArrayDeque<>();
    history.addLast(userMessage);
    history.addLast(assistantMessage);
    while (history.size() > MAX_MESSAGES) {
        history.removeFirst();  // 删除最早的
    }
    """)


# ============================================================================
# 第 5 节：SimpleChatEngine — 最简单的对话
# ============================================================================

def demo_simple_chat_engine():
    """
    SimpleChatEngine 不调用知识库，只是纯粹的对话

    适用于：闲聊、问答不涉及知识库的场景
    """
    print("=" * 60)
    print("【SimpleChatEngine — 最简单的对话】")
    print("=" * 60)

    from llama_index.core.chat_engine import SimpleChatEngine

    print("\n  --- 创建 SimpleChatEngine ---")
    print("  这个 Chat Engine 不调用知识库，只是跟 LLM 聊天")

    # 检查 LLM 是否配置
    if Settings.llm is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            from llama_index.llms.dashscope import DashScope
            Settings.llm = DashScope(
                model_name="qwen-plus",
                api_key=api_key,
                temperature=0.7,  # 闲聊可以用更高的温度
            )
            print(f"    ✓ 已配置 LLM: {Settings.llm.model_name}")
        else:
            print("    ⚠️  缺少 API Key，跳过演示\n")
            return

    chat_engine = SimpleChatEngine(streaming=False)

    print("\n  --- 闲聊演示 ---")
    print("  用户: 你好，介绍一下你自己")
    response = chat_engine.chat("你好，介绍一下你自己")
    print(f"  AI: {response.response[:100]}...")

    print("\n  用户: Python 和 Java 有什么区别？")
    response = chat_engine.chat("Python 和 Java 有什么区别？")
    print(f"  AI: {response.response[:100]}...")


# ============================================================================
# 第 6 节：三种 Chat Engine 对比
# ============================================================================

def demo_chat_engine_comparison():
    """
    三种 Chat Engine 的对比，帮你选择适合的
    """
    print("=" * 60)
    print("【Chat Engine 对比表】")
    print("=" * 60)

    print("""
  ┌──────────────────────────┬────────────┬──────────┬──────────────┐
  │ Chat Engine              │ 速度       │ 准确度   │ 适用场景      │
  ├──────────────────────────┼────────────┼──────────┼──────────────┤
  │ SimpleChatEngine         │ ★★★★★    │  ★★★    │ 闲聊/通用问答 │
  │ ConversableQueryEngine   │ ★★★★     │  ★★★★  │ RAG 对话     │
  │ ContextChatEngine        │ ★★★      │  ★★★★★ │ 精准 RAG 对话 │
  └──────────────────────────┴────────────┴──────────┴──────────────┘

  选择建议：
    - 只是想跟 LLM 聊天 → SimpleChatEngine
    - 需要基于知识库对话 → ConversableQueryEngineChatEngine（推荐）
    - 需要最高准确度 → ContextChatEngine（condense_plus_context）

  类比 Java：
    SimpleChatEngine       → 普通的 Service 方法
    ConversableQueryEngine → Service + Cache
    ContextChatEngine      → Service + 每次查 DB
    """)


# ============================================================================
# 第 7 节：完整示例 — 带记忆的 RAG 对话系统
# ============================================================================

def demo_complete_chat_system():
    """
    完整示例：构建一个带记忆的 RAG 对话系统

    流程：
    1. 准备知识库文档
    2. 构建 Index
    3. 创建 Chat Engine
    4. 进行多轮对话
    """
    print("=" * 60)
    print("【完整示例：带记忆的 RAG 对话系统】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.chat_engine import ConversableQueryEngineChatEngine

    # 第 1 步：准备知识库
    print("\n  --- 第 1 步：准备知识库 ---")
    documents = [
        Document(text="公司实行每日八小时工作制。上午9:00-12:00，下午13:30-17:30。"),
        Document(text="员工每年享有5天带薪年假，满10年10天，满20年15天。"),
        Document(text="病假需提供二级以上医院诊断证明，按基本工资80%发放。"),
        Document(text="迟到超过30分钟按旷工半天处理。每月有3次豁免。"),
        Document(text="入职缴纳五险一金，公积金个人12%公司12%。"),
        Document(text="年终奖根据个人绩效：A级6个月，B级3个月，C级1个月。"),
    ]
    print(f"    准备了 {len(documents)} 个知识库条目")

    # 第 2 步：构建 Index
    print("\n  --- 第 2 步：构建 Index ---")
    index = VectorStoreIndex.from_documents(documents)
    print("    ✓ Index 构建完成")

    # 第 3 步：创建 Chat Engine
    print("\n  --- 第 3 步：创建 Chat Engine ---")
    chat_engine = ConversableQueryEngineChatEngine(
        index=index,
        token_limit=4096,
    )
    print("    ✓ Chat Engine 创建完成")

    # 第 4 步：多轮对话
    print("\n  --- 第 4 步：多轮对话 ---")
    questions = [
        "公司的福利怎么样？",
        "年假具体怎么算？",
        "那满20年呢？",
        "迟到了怎么办？",
    ]

    for question in questions:
        print(f"\n  用户: {question}")
        response = chat_engine.chat(question)
        print(f"  AI: {response.response}")


# ============================================================================
# 第 8 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 9 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. Chat Engine 让 LlamaIndex 支持多轮对话                              │
│  2. Memory 存储对话历史，让 AI 记住上下文                                │
│  3. 三种 Chat Engine：Simple / Conversable / Context                    │
│  4. ConversableQueryEngineChatEngine 最常用                              │
│  5. ContextChatEngine 最准确但最慢                                      │
│  6. Memory 有 token 限制，防止无限增长                                  │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 创建带记忆的 RAG 对话系统                                             │
│  from llama_index.core.chat_engine import ConversableQueryEngineChatEngine│
│                                                                       │
│  index = VectorStoreIndex.from_documents(documents)                    │
│  chat_engine = ConversableQueryEngineChatEngine(                        │
│      index=index,                                                      │
│      token_limit=4096,                                                 │
│  )                                                                     │
│                                                                       │
│  # 多轮对话                                                             │
│  response = chat_engine.chat("第一个问题")                              │
│  response = chat_engine.chat("第二个问题（可以引用上一个问题）")         │
│  print(response.response)                                              │
│                                                                         │
│  【下一课预告】                                                           │
│  第 10 课：Agent + Tools / 智能体与工具                                  │
│  类比 Java：策略模式 + 工厂模式                                         │
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
║   第 9 课：Chat Engine + Memory / 对话引擎与记忆          ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. 为什么需要 Chat Engine？                             ║
║   2. ConversableQueryEngineChatEngine                    ║
║   3. ContextChatEngine                                   ║
║   4. Memory — 对话记忆详解                               ║
║   5. SimpleChatEngine                                    ║
║   6. 三种 Chat Engine 对比                               ║
║   7. 完整示例：带记忆的 RAG 对话系统                      ║
║   8. 总结                                               ║
║                                                          ║
║   前置知识：第 1-8 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：为什么需要 Chat Engine？ ━━━")
    demo_why_chat_engine()

    print("\n━━━ 第 2 节：ConversableQueryEngineChatEngine ━━━")
    demo_conversable_chat_engine()

    print("\n━━━ 第 3 节：ContextChatEngine ━━━")
    demo_context_chat_engine()

    print("\n━━━ 第 4 节：Memory 详解 ━━━")
    demo_memory()

    print("\n━━━ 第 5 节：SimpleChatEngine ━━━")
    demo_simple_chat_engine()

    print("\n━━━ 第 6 节：Chat Engine 对比 ━━━")
    demo_chat_engine_comparison()

    print("\n━━━ 第 7 节：完整示例 ━━━")
    demo_complete_chat_system()

    print("\n🎉 第 9 课完成！")
    print("   建议下一步：阅读 week03/code/p31-chat-engine.ipynb")
