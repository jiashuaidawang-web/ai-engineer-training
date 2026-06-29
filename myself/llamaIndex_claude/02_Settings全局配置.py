"""
===============================================================================
 第 2 课：Settings — LlamaIndex 的全局配置中心
===============================================================================

【这一课学什么？】
  在 LlamaIndex 中，几乎所有组件都需要配置：LLM 模型、Embedding 模型、
  分块大小、回调函数……如果每个地方都手动传参，代码会变得极其繁琐。
  Settings 就是解决方案 —— 一个全局单例，所有组件自动从中读取配置。

【类比 Java】
  Settings ≈ Spring 的 ApplicationContext / application.yml
  - 你在 application.yml 里配一次 datasource，所有 Repository 都能用
  - 你在 Settings 里配一次 llm 和 embed_model，所有 Index / Retriever / QueryEngine 都能用

【核心概念】
  Settings 是一个全局单例对象（类似 Java 的 Singleton），持有以下配置：
    - llm              → 语言模型（如 Qwen、GPT-4）
    - embed_model      → 向量嵌入模型（如 DashScope embedding）
    - chunk_size       → 默认分块大小
    - chunk_overlap    → 默认重叠大小
    - num_output       → LLM 生成回答的最大 token 数
    - context_window   → LLM 的上下文窗口大小
    - callback_manager → 追踪/日志回调

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/02_Settings全局配置.py

【前置知识】
  - 第 1 课：Node / 文本切片（已经学过了）
  - 需要你有 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量
"""

import os
from llama_index.core import Settings


# ============================================================================
# 第 1 节：Settings 是什么？为什么需要它？
# ============================================================================

def demo_why_settings():
    """
    演示：如果没有 Settings，代码会多麻烦

    假设我们不使用 Settings，每次创建组件都要手动传入配置：
    """
    print("=" * 60)
    print("【对比】不使用 Settings 的情况（伪代码，展示为什么麻烦）")
    print("=" * 60)

    # 想象一下，每个类都要手动传参：
    print("""
    # 第 1 步：创建 LLM
    llm = DashScopeLLM(model_name="qwen-max", api_key="sk-xxx", temperature=0.1)

    # 第 2 步：创建 Embedding 模型
    embed_model = DashScopeEmbedding(model_name="text-embedding-v3", api_key="sk-xxx")

    # 第 3 步：创建 NodeParser，要传 chunk_size
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

    # 第 4 步：创建 VectorStore
    vector_store = FAISS()

    # 第 5 步：创建 StorageContext，要传 vector_store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 第 6 步：创建 Index，要传 llm + embed_model + storage_context
    index = VectorStoreIndex.from_documents(
        documents=docs,
        llm=llm,          # ← 又要传一遍
        embed_model=embed_model,  # ← 又要传一遍
        storage_context=storage_context
    )

    # 第 7 步：创建 QueryEngine，还要再传一遍 llm！
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        llm=llm,          # ← 第 3 次传 llm 了...
        embed_model=embed_model  # ← 第 3 次传 embed_model 了...
    )
    """)

    print("\n看到了吗？llm 和 embed_model 被传递了 3 次以上！")
    print("如果还有 10 个组件，就要传 30 次... 这就是 Settings 存在的意义。\n")


# ============================================================================
# 第 2 节：Settings 的核心属性（类比 Java Bean 的 getter）
# ============================================================================

def demo_settings_attributes():
    """
    展示 Settings 的所有重要属性

    每个属性就像 Java Bean 的 getter：
      Settings.llm       → 获取配置的 LLM 对象
      Settings.embed_model → 获取配置的 Embedding 模型对象
      Settings.chunk_size → 获取默认分块大小
    """
    print("=" * 60)
    print("【Settings 属性一览】")
    print("=" * 60)

    # 查看所有可用属性（用 dir() 列出对象的所有方法和属性）
    # 类比 Java: Arrays.toString(MyClass.class.getDeclaredFields())
    attrs = [attr for attr in dir(Settings) if not attr.startswith('_')]
    print(f"\nSettings 共有 {len(attrs)} 个公开属性/方法：")
    for i, attr in enumerate(attrs, 1):
        print(f"  {i:2d}. {attr}")

    # 最常用的几个属性
    print("\n--- 最常用的属性 ---")
    print(f"  Settings._llm              = {Settings._llm}")
    print(f"  Settings._embed_model      = {Settings._embed_model}")
    print(f"  Settings._chunk_size       = {Settings._chunk_size}")
    print(f"  Settings._chunk_overlap    = {Settings._chunk_overlap}")
    print(f"  Settings._num_output       = {Settings._num_output}")
    print(f"  Settings._context_window   = {Settings._context_window}")


# ============================================================================
# 第 3 节：配置 LLM 模型（最核心的设置）
# ============================================================================

def demo_configure_llm():
    """
    演示如何配置 LLM 模型

    LlamaIndex 支持多种 LLM 后端：
      - DashScope（阿里云通义千问）
      - OpenAI（GPT-3.5 / GPT-4）
      - Ollama（本地部署的开源模型）
      - Anyscale / Azure / Bedrock 等

    这里以 DashScope（通义千问）为例，因为国内访问更快。
    """
    print("=" * 60)
    print("【配置 LLM：通义千问】")
    print("=" * 60)

    # 导入 DashScope LLM 适配器
    from llama_index.llms.dashscope import DashScope

    # 检查 API Key 是否已设置
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("  ⚠️  未找到 DASHSCOPE_API_KEY 环境变量")
        print("  请先执行: export DASHSCOPE_API_KEY='your-key-here'")
        print("  或者使用 OpenAI 兼容的 API Key\n")
        return

    # 创建 LLM 实例
    # 参数说明：
    #   model_name    → 模型名称（qwen-plus 性价比高，qwen-turbo 最快）
    #   api_key       → 阿里云 DashScope 的 API Key
    #   temperature   → 随机性（0=最确定，1=最创意，0.1=适合问答）
    llm = DashScope(
        model_name="qwen-plus",    # 模型：qwen-turbo(快) / qwen-plus(平衡) / qwen-max(强)
        api_key=api_key,           # API 密钥
        temperature=0.1,           # 低温度 = 更稳定、更准确的回答
    )

    # 将 LLM 设置到全局 Settings 中
    # 设置后，所有后续创建的 Index / QueryEngine 都会自动使用这个 LLM
    Settings.llm = llm

    print(f"  已配置 LLM: {llm.model_name}")
    print(f"  温度: {llm.temperature}")
    print(f"  Settings.llm 现在指向: {Settings.llm}")

    # 测试 LLM 是否能正常工作（发一个简单的请求）
    print("\n  --- 测试 LLM 连接 ---")
    try:
        # complete() = 发送文本给 LLM，让它生成回复
        # 类比 Java: llm.complete("你好") → "你好！有什么我可以帮你的？"
        response = llm.complete("你是谁？请用一句话回答。")
        print(f"  LLM 回答: {response.text}")
    except Exception as e:
        print(f"  ❌ 请求失败: {e}")


# ============================================================================
# 第 4 节：配置 Embedding 模型
# ============================================================================

def demo_configure_embedding():
    """
    演示如何配置 Embedding 模型

    Embedding 模型的作用：
      - 把文本转换成向量（一串浮点数）
      - 向量相似度 ≈ 语义相似度
      - 例如："猫" 和 "狗" 的向量距离很近，"猫" 和 "汽车" 的距离很远

    常用 Embedding 模型：
      - DashScope text-embedding-v3（中文效果好）
      - OpenAI text-embedding-ada-002（英文效果好）
      - BGE-M3（开源，可本地部署）
    """
    print("=" * 60)
    print("【配置 Embedding 模型】")
    print("=" * 60)

    # 检查 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("  ⚠️  未找到 DASHSCOPE_API_KEY 环境变量，跳过\n")
        return

    # 导入 DashScope Embedding 适配器
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    # 创建 Embedding 模型实例
    # 参数说明：
    #   model_name       → 模型名称
    #   api_key          → API 密钥
    #   embed_batch_size → 每次批量请求的文本数量（越大越快，但可能超时）
    #   embed_input_length → 每个文本的最大 token 数
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=api_key,
        embed_batch_size=10,            # 每次并发请求 10 条文本的 embedding
        embed_input_length=8192,        # 每条文本最多 8192 个 token
    )

    # 将 Embedding 模型设置到全局 Settings 中
    Settings.embed_model = embed_model

    print(f"  已配置 Embedding: {embed_model.model_name}")
    print(f"  Settings.embed_model 现在指向: {Settings.embed_model}")

    # 测试 Embedding 模型（把一句话转成向量）
    print("\n  --- 测试 Embedding 模型 ---")
    try:
        # get_text_embedding() = 把一段文本变成向量
        # 返回值是一个浮点数列表，如 [0.123, -0.456, 0.789, ...]
        # 类比 Java: List<Float> vec = embedModel.encode("你好世界");
        vector = embed_model.get_text_embedding("你好世界")
        print(f"  文本: '你好世界'")
        print(f"  向量维度: {len(vector)} 维")
        print(f"  向量前 10 个值: {[round(v, 4) for v in vector[:10]]}")
    except Exception as e:
        print(f"  ❌ 请求失败: {e}")


# ============================================================================
# 第 5 节：配置其他 Settings 参数
# ============================================================================

def demo_other_settings():
    """
    演示 Settings 的其他常用参数

    这些参数影响 LLM 的行为和性能：
    """
    print("=" * 60)
    print("【其他 Settings 参数配置】")
    print("=" * 60)

    # chunk_size: 文本切块的大小（以 token 为单位）
    # 类比 Java: 设置分片大小，就像 Kafka 的 max.message.bytes
    Settings.chunk_size = 1024
    print(f"  chunk_size       = {Settings.chunk_size}  （每个 Node 最多 {Settings.chunk_size} 个 token）")

    # chunk_overlap: 相邻块之间的重叠 token 数
    # 为什么需要重叠？防止关键信息刚好在两块边界处被切断
    # 类比 Java: 滑动窗口的 slide 距离，重叠 = window_size - slide
    Settings.chunk_overlap = 50
    print(f"  chunk_overlap    = {Settings.chunk_overlap}  （相邻块重叠 {Settings.chunk_overlap} 个 token）")

    # num_output: LLM 生成回答时最多输出的 token 数
    # 类比 Java: 设置返回结果的最大长度
    Settings.num_output = 256
    print(f"  num_output       = {Settings.num_output}  （回答最多 {Settings.num_output} 个 token）")

    # context_window: LLM 的上下文窗口大小
    # 即 LLM 一次能"记住"多少内容
    # qwen-plus 是 131072 token（约 10 万字），gpt-4 是 128000 token
    Settings.context_window = 128000
    print(f"  context_window   = {Settings.context_window}  （LLM 一次最多处理 {Settings.context_window} 个 token）")

    print("\n  💡 这些设置一旦配置，后续所有组件都会自动使用这些默认值！")


# ============================================================================
# 第 6 节：多 LLM / 多 Embedding 切换演示
# ============================================================================

def demo_switch_models():
    """
    演示如何在不同模型之间切换

    这在开发过程中非常有用：
    - 开发时用便宜的模型（qwen-turbo）
    - 生产时用强的模型（qwen-max）
    - 或者快速切换 OpenAI ↔ DashScope
    """
    print("=" * 60)
    print("【模型切换演示】")
    print("=" * 60)

    from llama_index.llms.dashscope import DashScope
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    # 保存原始设置（方便后面恢复）
    original_llm = Settings.llm
    original_embed = Settings.embed_model

    # 切换 1：使用更快的模型
    print("\n  --- 切换到 qwen-turbo（速度快、成本低）---")
    Settings.llm = DashScope(
        model_name="qwen-turbo",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        temperature=0.1,
    )
    print(f"  当前 LLM: {Settings.llm.model_name}")

    # 切换 2：使用更强的模型
    print("\n  --- 切换到 qwen-max（质量最好）---")
    Settings.llm = DashScope(
        model_name="qwen-max",
        api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        temperature=0.1,
    )
    print(f"  当前 LLM: {Settings.llm.model_name}")

    # 切换回原来的
    if original_llm:
        Settings.llm = original_llm
        print(f"\n  --- 恢复原始 LLM: {Settings.llm.model_name if hasattr(Settings.llm, 'model_name') else 'default'} ---")


# ============================================================================
# 第 7 节：Settings 与 Java 配置体系的完整对照
# ============================================================================

def demo_java_comparison():
    """
    用 Java 的配置体系来类比 LlamaIndex 的 Settings

    这样可以帮助 Java 程序员更好地理解 Settings 在整个架构中的位置。
    """
    print("=" * 60)
    print("【Settings vs Java 配置体系对照表】")
    print("=" * 60)

    print("""
  ┌──────────────────────────┬──────────────────────────────┬──────────────────────────┐
  │ LlamaIndex / Python      │ Java 等价物                  │ 说明                     │
  ├──────────────────────────┼──────────────────────────────┼──────────────────────────┤
  │ Settings.llm             │ @Bean LlmConfig              │ 全局 LLM 配置            │
  │ Settings.embed_model     │ @Bean EmbeddingConfig        │ 全局 Embedding 配置      │
  │ Settings.chunk_size      │ application.yml: chunk.size  │ 全局分块大小             │
  │ Settings.context_window  │ application.yml: context.window │ LLM 上下文窗口上限    │
  │ Settings.callback_manager│ SLF4J Logger / Micrometer    │ 全局日志/监控配置        │
  │                          │                              │                          │
  │ Settings 全局单例         │ ApplicationContext           │ Spring 容器              │
  │ from_documents()          │ @Autowired + 构造函数注入    │ Spring 依赖注入          │
  │ VectorStoreIndex          │ JpaRepository                │ 数据访问层               │
  │ StorageContext            │ DataSourceConfig             │ 数据库连接配置           │
  └──────────────────────────┴──────────────────────────────┴──────────────────────────┘
    """)


# ============================================================================
# 第 8 节：常见错误与调试技巧
# ============================================================================

def demo_debug_settings():
    """
    演示如何排查 Settings 配置问题

    常见问题：
    1. API Key 没设置 → 检查 os.environ
    2. 模型名称写错   → 查看 llama_index 文档确认
    3. 网络不通       → 检查代理设置
    4. 配额超限       → 检查 DashScope/OpenAI 控制台
    """
    print("=" * 60)
    print("【Settings 调试技巧】")
    print("=" * 60)

    print("""
  技巧 1：打印当前 Settings 的所有值
    from llama_index.core import Settings
    print(f"LLM: {Settings.llm}")
    print(f"Embed: {Settings.embed_model}")
    print(f"Chunk size: {Settings.chunk_size}")

  技巧 2：检查环境变量
    import os
    print(f"DASHSCOPE_API_KEY 存在? {'✓' if os.environ.get('DASHSCOPE_API_KEY') else '✗'}")
    print(f"OPENAI_API_KEY 存在?   {'✓' if os.environ.get('OPENAI_API_KEY') else '✗'}")

  技巧 3：用 try/except 捕获配置错误
    try:
        response = llm.complete("测试")
        print(f"成功: {response.text}")
    except Exception as e:
        print(f"配置错误: {type(e).__name__}: {e}")

  技巧 4：验证向量维度是否正确
    embed = Settings.embed_model
    vec = embed.get_text_embedding("测试")
    print(f"向量维度: {len(vec)}")  # DashScope v3 通常是 1024 或 1536
    """)


# ============================================================================
# 第 9 节：完整示例 — 配置好 Settings 后构建第一个 Index
# ============================================================================

def demo_settings_then_index():
    """
    完整示例：配置 Settings → 创建文档 → 构建 Index → 查询

    这是第 2 课的终极练习，把 Settings 和之前的 Node 知识串起来。
    虽然还没学到 Index（第 5 课），但可以先感受"配置 → 使用"的流程。
    """
    print("=" * 60)
    print("【完整示例：Settings + Document → Index → 查询】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core.indices.vector_store import VectorStoreIndex
    import faiss

    # 检查 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("  ⚠️  缺少 API Key，跳过完整示例\n")
        return

    # 第 1 步：配置 Settings
    print("\n  第 1 步：配置 Settings...")
    from llama_index.llms.dashscope import DashScope
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    Settings.llm = DashScope(
        model_name="qwen-plus",
        api_key=api_key,
        temperature=0.1,
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=api_key,
        embed_batch_size=10,
    )
    print(f"    ✓ LLM: {Settings.llm.model_name}")
    print(f"    ✓ Embedding: {Settings.embed_model.model_name}")

    # 第 2 步：创建一些文档（用第 1 课学过的知识）
    print("\n  第 2 步：创建文档...")
    docs_text = """
    公司实行每日八小时工作制。
    上午工作时间为九时至十二时，中午休息一小时。
    下午工作时间为十三时三十分至十七时三十分。
    员工每周享有两天休息日，通常为周六和周日。
    因工作需要安排加班的，应优先安排补休。
    无法安排补休的，按国家规定支付加班工资。
    """
    doc = Document(text=docs_text, metadata={"title": "考勤制度"})
    print(f"    ✓ 创建了 1 个文档，{len(doc.text)} 字符")

    # 第 3 步：用 Settings 中自动获取的 LLM 和 Embedding 构建 Index
    print("\n  第 3 步：构建 VectorStoreIndex...")
    # 注意：这里没有传 llm 或 embed_model！因为它们已经在 Settings 中配置好了
    # 类比 Java: 不需要在每个 Repository 构造函数里传 DataSource，
    #           因为 Spring 会自动从 ApplicationContext 注入
    index = VectorStoreIndex.from_documents([doc])
    print(f"    ✓ Index 构建完成，包含 {len(index.docstore.docs)} 个文档")

    # 第 4 步：使用 Index 查询
    print("\n  第 4 步：查询 Index...")
    query_engine = index.as_query_engine()
    response = query_engine.query("员工每天工作几个小时？")
    print(f"    问: 员工每天工作几个小时？")
    print(f"    答: {response.response}")

    print("\n  🎉 恭喜！你刚刚用 Settings 配置 + Index 完成了一次完整的 RAG 查询！")
    print("  后续课程会深入讲解每个组件的细节。")


# ============================================================================
# 第 10 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 2 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. Settings 是 LlamaIndex 的全局配置中心                                 │
│  2. 配置一次，所有组件自动使用（LLM、Embedding、分块参数等）              │
│  3. 类比 Java 的 ApplicationContext / application.yml                   │
│  4. 支持的 LLM：DashScope / OpenAI / Ollama 等                           │
│  5. 支持的 Embedding：DashScope / OpenAI / BGE 等                        │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 配置 LLM                                                              │
│  from llama_index.llms.dashscope import DashScope                       │
│  Settings.llm = DashScope(model_name="qwen-plus", api_key="sk-xxx")     │
│                                                                         │
│  # 配置 Embedding                                                        │
│  from llama_index.embeddings.dashscope import DashScopeEmbedding        │
│  Settings.embed_model = DashScopeEmbedding(                               │
│      model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,          │
│      api_key="sk-xxx"                                                    │
│  )                                                                      │
│                                                                         │
│  # 配置其他参数                                                          │
│  Settings.chunk_size = 1024                                             │
│  Settings.chunk_overlap = 50                                            │
│  Settings.num_output = 256                                              │
│                                                                         │
│  【下一课预告】                                                           │
│  第 3 课：Reader / 文档加载 — 从文件夹读取 PDF、TXT、DOCX 等文件         │
│  类比 Java：Apache Tika / Apache POI                                    │
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
║   第 2 课：Settings — LlamaIndex 的全局配置中心           ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. 为什么需要 Settings？                                ║
║   2. Settings 的核心属性                                  ║
║   3. 配置 LLM 模型                                       ║
║   4. 配置 Embedding 模型                                  ║
║   5. 其他 Settings 参数                                  ║
║   6. 多模型切换演示                                      ║
║   7. Settings vs Java 配置体系对照                        ║
║   8. 调试技巧                                            ║
║   9. 完整示例：Settings → Index → Query                   ║
║   10. 总结                                               ║
║                                                          ║
║   前置知识：第 1 课 Node / 文本切片                        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：为什么需要 Settings？ ━━━")
    demo_why_settings()

    print("\n━━━ 第 2 节：Settings 属性一览 ━━━")
    demo_settings_attributes()

    print("\n━━━ 第 3 节：配置 LLM 模型 ━━━")
    demo_configure_llm()

    print("\n━━━ 第 4 节：配置 Embedding 模型 ━━━")
    demo_configure_embedding()

    print("\n━━━ 第 5 节：其他 Settings 参数 ━━━")
    demo_other_settings()

    print("\n━━━ 第 6 节：模型切换演示 ━━━")
    demo_switch_models()

    print("\n━━━ 第 7 节：Settings vs Java 对照 ━━━")
    demo_java_comparison()

    print("\n━━━ 第 8 节：调试技巧 ━━━")
    demo_debug_settings()

    print("\n━━━ 第 9 节：完整示例 ━━━")
    demo_settings_then_index()

    print("\n🎉 第 2 课完成！")
    print("   建议下一步：阅读 week03/code/p24-settings.ipynb")
