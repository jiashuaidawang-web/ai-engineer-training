"""
05_rag_demo.py — Milvus + LLM 实战：构建 RAG 知识库

本课将 Milvus 集成到一个完整的 RAG（检索增强生成）系统中。

RAG 架构：
┌──────────┐    文本    ┌──────────────┐    向量    ┌──────────┐
│ 原始文档  │ ────────→ │  Chunk 切分   │ ────────→ │ Embedding│
└──────────┘           └──────────────┘            └────┬─────┘
                                                        ↓
┌──────────┐    回答    ┌──────────────┐    向量    ┌──────────┐
│  LLM 回答 │ ←──────── │  上下文组装    │ ←──────── │ Milvus   │
└──────────┘           └──────────────┘            └──────────┘
                                ↑                        ↑
                           用户问题                  向量检索

本 Demo 模拟完整流程：
  1. 文档入库：模拟文档 → 切块 → 向量化 → 存入 Milvus
  2. 检索增强：用户问题 → 向量化 → Milvus 检索 Top-K
  3. 生成回答：检索结果 + 用户问题 → 组装 Prompt → LLM 生成回答

【Java 程序员速查】
  这个文件引入了新的 import 语法：
    from typing import List, Dict
  Python 3.9+ 可以直接用 list[dict]，但 typing 模块提供旧版本的类型注解
  类比 Java: List<String>, Map<String, Object>
"""

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
import numpy as np
import re       # 正则表达式 — 类比 Java: java.util.regex.Pattern
import hashlib  # 哈希工具 — 类比 Java: java.security.MessageDigest
import warnings
warnings.filterwarnings("ignore")

connections.connect("default", host="localhost", port="19530")


# ============================================================
# 第一步：模拟 Embedding（实际项目替换为真实模型）
# ============================================================

def mock_embedding(text: str, dim: int = 128) -> list:
    """
    模拟 Embedding 函数

    实际项目中，这里应该调用：
      - OpenAI: openai.Embedding.create(input=text, model="text-embedding-ada-002")
      - DashScope: dashscope.text_embedding.call(model="text-embedding-v3", inputs=[...])
      - 本地模型: sentence-transformers / bge-large-zh

    这里用哈希生成确定性向量（相同文本总是得到相同向量），
    保证 Demo 可重复运行。

    【Python 语法】类型注解（Type Hints）
      text: str      → 参数 text 的类型是 str（类比 Java: String text）
      dim: int = 128 → 参数 dim 类型是 int，默认值 128（类比 Java: int dim = 128）
      -> list[float] → 返回值类型是 float 列表（类比 Java: List<Float>）

      注意：Python 的类型注解是「软」的，运行时不会强制检查
      类比 Java 的 @SuppressWarnings("unchecked")，只是给 IDE 和开发者看的
    """
    h = hashlib.sha256(text.encode()).hexdigest()  # 生成 SHA-256 哈希
    np.random.seed(int(h[:8], 16))  # 用哈希的前8位作为随机种子
    return np.random.rand(dim).astype(np.float32).tolist()  # 转成 Python list


# ============================================================
# 第二步：文档切块（Chunking）
# ============================================================

def chunk_document(text: str, chunk_size: int = 200, overlap: int = 50) -> list:
    """
    将长文本切分为小块

    切块策略：
    1. 按段落分割
    2. 合并短段落，直到达到 chunk_size
    3. 相邻块之间有 overlap（重叠），保证语义连续性

    参数：
      text:       原始文本
      chunk_size: 每个块的字符数
      overlap:    相邻块之间的重叠字符数

    返回：
      [{"content": "...", "metadata": {...}}, ...]

    【Python 语法】f-string + 多行字符串
      f"chunk_size={chunk_size}" 类比 Java: "chunk_size=" + chunk_size
      多行字符串用 ''' 或 """ 包裹，类比 Java 的文本块 (Java 15+): """..."""
    """
    # --- 【Python 语法】re.split() ---
    # 按换行符分割文本
    # 类比 Java: text.split("\\n")
    paragraphs = re.split(r'[\n\r]+', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]  # 过滤空段落

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # --- 【Python 语法】re.split 保留分隔符 ---
            sentences = re.split(r'([。！？.!?])', para)
            sentence_parts = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    sentence_parts.append(sentences[i] + sentences[i+1])
                else:
                    sentence_parts.append(sentences[i])

            for sent in sentence_parts:
                if len(current_chunk) + len(sent) > chunk_size and current_chunk:
                    chunks.append({"content": current_chunk})
                    if overlap > 0:
                        current_chunk = current_chunk[-overlap:] + sent
                    else:
                        current_chunk = sent
                else:
                    current_chunk += sent
        else:
            if len(current_chunk) + len(para) + 1 > chunk_size and current_chunk:
                chunks.append({"content": current_chunk})
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:] + "\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n" + para if current_chunk else para

    if current_chunk:
        chunks.append({"content": current_chunk})

    return chunks


# ============================================================
# 第三步：构建知识库集合
# ============================================================

def build_knowledge_base():
    """
    构建 Milvus 知识库集合

    类比 Java 工厂方法:
      public KnowledgeBase buildKnowledgeBase() {
          CollectionSchema schema = ...;
          Collection collection = new Collection(schema);
          collection.createIndex(...);
          return collection;
      }
    """
    collection_name = "rag_knowledge_base"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    ]
    schema = CollectionSchema(fields, description="RAG 知识库")
    collection = Collection(collection_name, schema=schema)

    collection.create_index("embedding", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    })

    return collection


# ============================================================
# 第四步：模拟文档数据
# ============================================================

# --- 【Python 语法】原始字符串（Raw String）---
# r"""...""" 中的 r 表示「原始字符串」，\n 不会被转义为换行符
# 类比 Java: """...""" 文本块（Java 15+）
SAMPLE_DOCUMENTS = [
    {
        "title": "Milvus 简介",
        "source": "milvus_docs.pdf",
        "content": """
Milvus 是一个云原生向量数据库，由 Zilliz 开发。
它为 AI 应用中的海量向量提供索引和检索服务。

核心特性：
1. 千亿级向量支持：Milvus 采用存算分离架构，支持水平扩展，可管理超过十亿级的向量数据。
2. 混合搜索：同时支持稠密向量（Dense Vector）和稀疏向量（Sparse Vector）的混合检索。
3. 多种索引类型：提供 IVF_FLAT、IVF_SQ8、HNSW、SCANN、DISKANN 等多种索引算法。
4. 实时检索：支持实时数据插入和检索，数据写入后可立即被搜索到。
5. 高可用：支持多副本部署，自动故障转移。

应用场景：
- 语义搜索：文本、图片、音频的相似度搜索
- 推荐系统：基于用户行为的个性化推荐
- 异常检测：识别偏离正常模式的异常数据点
- RAG（检索增强生成）：为大语言模型提供外部知识来源
        """.strip(),
    },
    {
        "title": "向量索引算法",
        "source": "index_algorithms.pdf",
        "content": """
向量索引算法决定了如何高效地搜索相似向量。

FLAT（暴力搜索）:
- 计算查询向量与数据库中每一个向量的距离
- 结果绝对精确，但速度随数据量线性下降
- 适合数据量小于 10 万的场景

IVF_FLAT（倒排文件索引）:
- 先用 K-Means 将向量分成 K 个簇
- 搜索时先找到最近的几个簇，只在这些簇中计算距离
- nlist: 簇的数量，通常取 sqrt(N)，N 为数据总量
- nprobe: 搜索时考察的簇数，越大越精确但越慢

HNSW（分层导航图）:
- 构建多层图结构，上层跳跃式搜索，下层精细搜索
- 查询速度快，精度高，是目前最常用的索引
- M: 每个节点的最大连接数，通常 8~64
- efConstruction: 建图时的搜索宽度，越大图质量越高
- ef: 搜索时的搜索宽度，越大越精确但越慢

DISKANN:
- 专为磁盘优化的图索引
- 适合内存放不下全部向量的场景
- 比 HNSW 慢但能用磁盘存更多数据
        """.strip(),
    },
    {
        "title": "RAG 架构设计",
        "source": "rag_design.pdf",
        "content": """
RAG（Retrieval-Augmented Generation）= 检索 + 生成

为什么需要 RAG？
大语言模型（LLM）存在以下局限：
1. 知识截止：训练数据有截止日期，不知道新知识
2. 领域知识：通用模型缺乏垂直领域的专业知识
3. 幻觉：可能编造不存在的事实
4. 私有数据：无法访问企业的私有数据

RAG 如何解决：
1. 将私有文档切块、向量化、存入向量数据库
2. 用户提问时，先在向量库中检索相关文档
3. 将检索结果作为上下文提供给 LLM
4. LLM 基于检索到的内容生成准确回答

关键挑战：
- 文档切块：chunk size 太小丢失上下文，太大引入噪声
- 检索质量：Top-K 的选择，混合搜索 vs 纯向量搜索
- 重排序：用交叉编码器对检索结果重新排序
- 响应生成：如何让 LLM 基于检索内容诚实回答
        """.strip(),
    },
]


# ============================================================
# 第五步：入库流程
# ============================================================

def ingest_documents(collection, documents: list):
    """
    文档入库流程：
    文档 → 切块 → 向量化 → 批量插入 Milvus

    【Python 语法】参数类型注解
      documents: list[dict]  → 参数 documents 是 dict 列表
      类比 Java: List<Map<String, Object>> documents
    """
    all_embeddings = []
    all_contents = []
    all_sources = []
    all_chunk_ids = []
    all_titles = []

    chunk_counter = 0

    for doc in documents:
        # Step 1: 切块
        chunks = chunk_document(doc["content"], chunk_size=300, overlap=50)

        for chunk_idx, chunk in enumerate(chunks):
            # --- 【Python 语法】enumerate() ---
            # enumerate(chunks) 返回 (索引, 元素) 的对
            # 类比 Java: for (int i = 0; i < chunks.size(); i++) { var chunk = chunks.get(i); }
            embedding = mock_embedding(chunk["content"])

            all_embeddings.append(embedding)
            all_contents.append(chunk["content"])
            all_sources.append(doc["source"])
            all_chunk_ids.append(chunk_counter)
            all_titles.append(doc["title"])
            chunk_counter += 1

    print(f"[入库] 共 {len(all_contents)} 个文本块，{len(documents)} 篇文档")
    print(f"[入库] 正在插入 Milvus...")

    collection.insert([
        all_embeddings,
        all_contents,
        all_sources,
        all_chunk_ids,
        all_titles,
    ])

    collection.load()
    print(f"[入库] OK 完成，共 {collection.num_entities} 条向量数据\n")

    return chunk_counter


# ============================================================
# 第六步：检索 + 生成（RAG 核心流程）
# ============================================================

def rag_query(collection, question: str, top_k: int = 3):
    """
    RAG 查询流程：
    1. 将问题向量化
    2. 在 Milvus 中检索最相关的 Top-K 文档块
    3. 组装 Prompt 发送给 LLM
    4. 返回生成结果

    【Python 语法】默认参数
      question: str, top_k: int = 3
      类比 Java 的方法重载:
        void ragQuery(Collection coll, String question) {
            ragQuery(coll, question, 3);
        }
        void ragQuery(Collection coll, String question, int topK) { ... }
    """
    print("=" * 60)
    print(f"  RAG 查询: {question}")
    print("=" * 60)

    # Step 1: 问题向量化
    question_embedding = mock_embedding(question)

    # Step 2: Milvus 检索
    print(f"\n[检索] 正在检索 Top-{top_k} 相关文档块...")
    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 64},
        },
        limit=top_k,
        output_fields=["content", "source", "title"],
    )

    # Step 3: 展示检索结果
    print(f"\n[检索] 找到 {len(results[0])} 个相关块:")
    context_parts = []
    for i, hit in enumerate(results[0]):
        title = hit.fields.get("title", "Unknown")
        source = hit.fields.get("source", "Unknown")
        content = hit.fields.get("content", "")[:150]  # 截断显示
        distance = hit.distance

        context_parts.append(content)

        print(f"  #{i+1} [得分:{1-distance:.3f}] {title}")
        print(f"      来源: {source}")
        print(f"      内容: {content}...")
        print()

    # Step 4: 组装 LLM Prompt
    # --- 【Python 语法】join ---
    # "\n\n---\n\n".join(context_parts) 类比 Java:
    #   String.join("\n\n---\n\n", contextParts.toArray(new String[0]))
    context = "\n\n---\n\n".join(context_parts)

    # --- 【Python 语法】f-string 多行拼接 ---
    # 类比 Java:
    #   String prompt = "参考资料：\n" + context + "\n\n问题：" + question;
    prompt = f"""你是一个有帮助的助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{question}

请基于参考资料给出准确的回答。如果参考资料中没有相关信息，请如实告知。"""

    print("=" * 60)
    print("  模拟 LLM 回答（实际项目中替换为真实 API 调用）")
    print("=" * 60)
    print(f"\n[Prompt] 长度: {len(prompt)} 字符")
    print(f"\n[回答] （此处应调用 OpenAI/DashScope 等 LLM API）")
    print(f"       参考资料数: {len(context_parts)}")
    print(f"       用户问题: {question}")

    return {
        "question": question,
        "contexts": context_parts,
        "prompt": prompt,
    }


# ============================================================
# 第七步：带过滤的 RAG 查询
# ============================================================

def rag_query_with_filter(collection, question: str, source_filter: str, top_k: int = 3):
    """
    带来源过滤的 RAG 查询

    【Python 语法】字符串格式化
      f"source == '{source_filter}'" 类比 Java:
      "source == '" + sourceFilter + "'"
    """
    question_embedding = mock_embedding(question)
    filter_expr = f"source == '{source_filter}'"

    print(f"\n[过滤检索] 来源过滤: {filter_expr}")
    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        expr=filter_expr,  # ← 标量过滤条件
        output_fields=["content", "source", "title"],
    )

    print(f"[过滤检索] 找到 {len(results[0])} 个相关块")
    for i, hit in enumerate(results[0]):
        print(f"  #{i+1} [{hit.fields.get('title', '?')}] 得分:{1-hit.distance:.3f}")

    return results


# ============================================================
# 主程序
# ============================================================

def main():
    print("Milvus + LLM 实战：RAG 知识库 Demo\n")

    # Step 1: 构建知识库
    print(">>> 构建知识库...")
    collection = build_knowledge_base()
    total_chunks = ingest_documents(collection, SAMPLE_DOCUMENTS)

    # Step 2: 测试 RAG 查询
    print(">>> 测试 RAG 查询...\n")
    rag_query(collection, "Milvus 支持哪些索引类型？")

    print("\n")
    rag_query(collection, "什么是 RAG 架构？为什么需要它？")

    print("\n")
    rag_query(collection, "向量索引 IVF 和 HNSW 有什么区别？")

    # Step 3: 带过滤的查询
    print("\n>>> 带来源过滤的查询...")
    rag_query_with_filter(collection, "Milvus 的特性", "milvus_docs.pdf")

    # 清理
    collection.drop()
    connections.disconnect("default")

    print("\n" + "=" * 60)
    print("  OK RAG Demo 完成！")
    print("=" * 60)
    print("""
下一步优化方向：
1. 接入真实 Embedding 模型（OpenAI / DashScope / 本地 BGE）
2. 接入真实 LLM API（OpenAI / Qwen / ChatGLM）
3. 加入重排序（Cross-Encoder Reranker）
4. 支持多格式文档（PDF / DOCX / Markdown）
5. 增量更新和去重
    """)


if __name__ == "__main__":
    main()
