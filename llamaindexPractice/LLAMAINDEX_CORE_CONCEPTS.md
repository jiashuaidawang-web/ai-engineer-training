# LlamaIndex 核心概念总结

## 一、架构概览

```
用户提问
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│                    Query Engine（查询引擎）                │
│              接收问题 → 调用检索 → 生成回答                │
└──────────────────────┬──────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   Retriever       │    │  Response        │
│   （检索器）       │    │  Synthesizer     │
│                  │    │  （响应合成器）     │
│ • 向量检索        │    │                  │
│ • 关键词检索       │    │ • CompactAndRefine│
│ • 混合检索        │    │ • Refine         │
│                  │    │ • TreeSummarize  │
└──────────┬───────┘    └──────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                    Vector Store（向量存储）                │
│              存储向量 + 元数据 + 索引                       │
└─────────────────────────────────────────────────────────┘
           ▲
           │
┌─────────────────────────────────────────────────────────┐
│                    Index（索引）                          │
│              组织文档 → 构建向量 → 存储                     │
└─────────────────────────────────────────────────────────┘
```

## 二、核心概念速查

### 数据模型

| 概念 | 说明 | 类比 |
|------|------|------|
| Document | 原始文档（文本 + 元数据） | 数据库中的一行记录 |
| Node | 文档的分块（Chunk） | 文档的子片段 |
| Index | 索引结构 | 数据库索引 |
| Retriever | 检索器（从索引中查找） | 数据库查询 |
| QueryEngine | 查询引擎（完整问答流程） | Service 层 |
| ChatEngine | 对话引擎（多轮对话） | 带状态的 Service |

### 索引类型

| 索引 | 适用场景 | 检索方式 |
|------|---------|---------|
| VectorStoreIndex | 语义搜索（最常用） | 向量相似度 |
| SummaryIndex | 全文搜索 | 全文匹配 |
| KeywordTableIndex | 关键词匹配 | 关键词倒排 |
| TreeIndex | 层次化检索 | 树遍历 |

### 响应合成策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| CompactAndRefine | 先压缩再精炼 | 默认推荐 |
| Refine | 逐文档生成部分回答 | 需要详细回答 |
| TreeSummarize | 树形摘要 | 长文档总结 |

## 三、常用 API 速查

### 基础流程
```python
from llama_index.core import Settings, Document, VectorStoreIndex

# 1. 配置模型
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key="your-key")
Settings.embed_model = OpenAIEmbedding(api_key="your-key")

# 2. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 3. 构建索引
index = VectorStoreIndex.from_documents(documents)

# 4. 创建查询引擎
query_engine = index.as_query_engine()

# 5. 提问
response = query_engine.query("你的问题")
print(response.response)
```

### 自定义检索
```python
# 创建检索器
retriever = index.as_retriever(
    similarity_top_k=5,      # 返回 Top-5
    similarity_cutoff=0.7,   # 相似度阈值
)

# 检索
results = retriever.retrieve("你的问题")
for result in results:
    print(f"相似度: {result.score:.4f}")
    print(f"文本: {result.text}")
```

### 自定义提示模板
```python
from llama_index.core.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    "你是一个专业的技术助手。请基于以下信息回答问题：\n"
    "{context_str}\n\n问题：{query_str}\n回答："
)

query_engine = index.as_query_engine(
    text_qa_template=custom_prompt,
)
```

### 对话引擎
```python
chat_engine = index.as_chat_engine(chat_mode="openai")
response = chat_engine.chat("你好，请问 LlamaIndex 是什么？")
print(response.response)
```

## 四、与 LangChain 对比

| 维度 | LlamaIndex | LangChain |
|------|-----------|-----------|
| 定位 | RAG 专家 | Agent 框架 |
| 优势 | 文档处理、检索、索引 | Agent、工具调用、多步推理 |
| 学习曲线 | 较平缓 | 较陡峭 |
| 适合场景 | 知识库、文档问答 | 自动化工作流、多步任务 |
| 灵活性 | 专注 RAG 场景 | 通用 AI 应用 |
| 社区 | 快速增长 | 成熟庞大 |

## 五、性能调优建议

### 文档加载
- 只加载需要的文件类型
- 使用增量加载避免重复处理
- 清理文档中的噪声（页眉、页脚、水印）

### 文本分块
- chunk_size: 中文 512-1024, 英文 768-1536
- chunk_overlap: chunk_size 的 10-20%
- 优先按语义边界分割（句子、段落）

### 检索调优
- similarity_top_k: 3-10（根据文档数量调整）
- similarity_cutoff: 0.5-0.8（过滤低相似度结果）
- 考虑混合搜索（向量 + 关键词）

### 响应合成
- CompactAndRefine: 默认推荐
- Refine: 需要更详细的回答
- TreeSummarize: 长文档总结

## 六、常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| 回答不准确 | 文档不包含相关信息 | 检查文档覆盖率 |
| 回答太慢 | 检索文档过多 | 减少 similarity_top_k |
| 回答幻觉 | 检索结果不相关 | 提高 similarity_cutoff |
| API 报错 | API Key 无效 | 检查 .env 配置 |
| 文档加载失败 | 格式不支持 | 检查文件格式和编码 |
| 内存不足 | 文档太大 | 减少 chunk_size 或使用流式加载 |
