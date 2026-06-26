# Milvus 向量数据库 — 从零到精通学习计划

> 目标：通过手写代码 + 原理解读，深入理解 Milvus 作为生产级向量数据库的核心能力。

## 学习路线

| 阶段 | 文件 | 内容 | 预计时间 |
|------|------|------|----------|
| 1 | `01_milvus_primer.py` | 环境准备 + 基础 CRUD | 30min |
| 2 | `02_vector_search.py` | 向量搜索 + 索引类型 | 40min |
| 3 | `03_filter_hybrid_search.py` | 标量过滤 + 混合搜索 | 30min |
| 4 | `04_advanced_features.py` | 分区、别名、一致性级别 | 30min |
| 5 | `05_rag_demo.py` | RAG 知识库实战 | 40min |

## 前置知识

### 什么是向量数据库？

传统数据库存的是「结构化数据」（行和列），而向量数据库存的是「向量」（高维数组）。

```
文本 "一只猫坐在沙发上" → 经过 Embedding 模型 → [0.12, -0.34, 0.87, ..., 0.45]  （1536维向量）
```

向量数据库的核心能力：**给定一个查询向量，快速找到数据库中最相似的向量**。

### 为什么选 Milvus？

| 特性 | Milvus | FAISS | Pinecone |
|------|--------|-------|----------|
| 开源 | ✅ | ✅ | ❌ |
| 分布式 | ✅ | ❌ | ✅ |
| 持久化 | ✅ | ❌ | ✅ |
| 标量过滤 | ✅ | 需自己实现 | ✅ |
| 混合搜索 | ✅ | ❌ | ✅ |
| 运维复杂度 | 中等 | 低 | 极低 |
| 适用场景 | 生产环境 | 实验/嵌入 | 不想运维 |

### Milvus 核心概念速查

| 概念 | 类比 | 说明 |
|------|------|------|
| **Collection** | 数据库中的表 | 存储向量数据的容器 |
| **Schema** | 表的字段定义 | 定义哪些字段存什么类型的数据 |
| **Partition** | 表的子分区 | 用于数据隔离和加速查询 |
| **Index** | 索引 | 加速向量搜索的数据结构（IVF_FLAT, HNSW, DISKANN...） |
| **Vector** | 一行数据中的向量子段 | 128~15360 维的高维数组 |
| **Distance** | 相似度度量 | L2（越小越相似）、Cosine（余弦相似度） |
| **Consistency Level** | 数据可见性 | Strong（强一致）、Bounded（有延迟）、Eventual（最终一致） |

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Milvus 服务（二选一）

# 方式A：Docker（推荐）
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v ./mydata/milvus:/var/lib/milvus \
  milvusdb/milvus:v2.5.0-beta \
  milvus run standalone

# 方式B：使用 Lite 版（无需 Docker，纯 Python，适合学习）
# 代码中已内置 Lite 模式自动降级

# 3. 按顺序运行代码文件
python 01_milvus_primer.py
python 02_vector_search.py
# ...
```

## 推荐配合阅读

- [CORE_CONCEPTS.md](CORE_CONCEPTS.md) — **核心概念速查表**（API对照、索引对比、性能调优、排错指南）← 必读！
- [Milvus 官方文档](https://milvus.io/docs) — 对照代码看文档，理解更快
- [Pymilvus API 参考](https://milvus.io/docs/reference/pymilvus/v2.5.x/about-pymilvus.md)
- [向量索引算法详解](https://milvus.io/docs/index.md) — IVF_FLAT / HNSW / SCANN 的区别

## 学习建议

1. **按顺序运行**：01 → 02 → 03 → 04 → 05，后面的代码依赖前面的概念
2. **边读边改**：把代码里的数字改一改（向量维度、数据量、索引参数），观察结果变化
3. **对照官方文档**：遇到不懂的 API，去 [pymilvus 参考](https://milvus.io/docs/reference/pymilvus/v2.5.x/about-pymilvus.md) 查
4. **先看原理再动手**：每个文件开头的「概念」注释块是精华，不要跳过
