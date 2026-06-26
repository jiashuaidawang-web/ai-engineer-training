# Milvus 核心概念总结

## 一、架构概览

```
                    ┌─────────────┐
                    │  pymilvus   │  Python SDK
                    │  (你的代码)  │
                    └──────┬──────┘
                           │ gRPC / HTTP
                    ┌──────▼──────┐
                    │  Proxy      │  请求路由 & 协调
                    └──┬───┬───┬──┘
                       │   │   │
              ┌────────▼┐ ┌▼───▼───────┐  ┌──────────────────┐
              │ QueryNode│ │ DeltaNode  │  │ DataNode         │
              │ (搜索计算)│ │ (删除记录)  │  │ (数据存储)       │
              └─────────┘ └────────────┘  └──────────────────┘
                       │
              ┌────────▼─────────┐
              │  etcd (元数据)    │  集合/Schema/索引信息
              └──────────────────┘
              ┌──────────────────┐
              │  MinIO / S3      │  持久化存储
              └──────────────────┘
```

## 二、核心概念速查

### 数据模型

| 概念 | 说明 | 类比 |
|------|------|------|
| Collection | 向量数据容器 | 数据库中的 Table |
| Schema | 集合的结构定义 | Table 的 Column 定义 |
| Partition | 集合内的逻辑分区 | Table 的 Partition |
| Entity | 一条数据记录 | Table 中的一行 |
| Vector | 高维向量（128~15360维） | 一行中的向量子段 |
| Alias | 集合的别名 | Table 的 Symlink |

### 向量类型

| 类型 | 说明 | 维度 |
|------|------|------|
| FLOAT_VECTOR | 稠密浮点向量（最常见） | 1~16384 |
| FLOAT16_VECTOR | 16位浮点量化向量 | 1~16384 |
| BF16_VECTOR | 脑浮点向量 | 1~16384 |
| SPARSE_FLOAT_VECTOR | 稀疏向量（关键词权重） | 词表大小 |
| SPARSE_BINARY_VECTOR | 稀疏二进制向量 | 词表大小 |

### 距离度量

| 度量 | 公式 | 范围 | 含义 |
|------|------|------|------|
| L2 | √Σ(xᵢ-yᵢ)² | [0, ∞) | 越小越相似 |
| COSINE | 1 - (A·B)/(|A||B|) | [0, 2] | 越小越相似 |
| IP | 1 - A·B | [-∞, 1] | 越小越相似 |
| HAMMING | 异或后1的个数 | [0, dim] | 越小越相似 |
| JACCARD | 异或/或 | [0, 1] | 越小越相似 |

### 索引类型

| 索引 | 类别 | 适用数据量 | 构建速度 | 查询速度 | 内存占用 |
|------|------|-----------|---------|---------|---------|
| FLAT | 暴力 | < 10万 | 快 | 慢 | 低 |
| IVF_FLAT | 聚类 | 10万~千万 | 中 | 快 | 中 |
| IVF_SQ8 | 量化聚类 | 10万~千万 | 中 | 快 | 低(75%节省) |
| HNSW | 图 | 百万~亿 | 慢 | 很快 | 高 |
| SCANN | 乘积量化 | 百万~千万 | 中 | 很快 | 中 |
| DISKANN | 磁盘图 | 千万~十亿 | 慢 | 很快 | 低(用磁盘) |
| GPU_IVF_FLAT | GPU聚类 | 亿级 | 中 | 极快 | GPU显存 |

### 一致性级别

| 级别 | 延迟 | 吞吐量 | 适用场景 |
|------|------|--------|---------|
| STRONG | 实时 | 低 | 金融、支付 |
| BOUNDED | 毫秒~秒 | 中 | 默认推荐 |
| SESSION | 同会话 | 高 | 社交动态 |
| EVENTUAL | 无保证 | 最高 | 推荐系统 |

## 三、常用 API 速查

### 连接
```python
from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
connections.disconnect("default")
```

### 集合操作
```python
from pymilvus import Collection, utility

# 检查集合是否存在
utility.has_collection("my_collection")

# 列出所有集合
utility.list_collections()

# 创建集合
collection = Collection(name="my_collection", schema=schema)

# 加载索引到内存
collection.load()

# 从内存释放
collection.release()

# 查看集合信息
collection.num_entities       # 实体数量
collection.partitions         # 分区列表
collection.indexes            # 索引列表

# 删除集合
utility.drop_collection("my_collection")
```

### 数据操作
```python
# 插入
collection.insert([data_list])

# 删除
collection.delete(expr="id in [1, 2, 3]")

# 搜索
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 16}},
    limit=10,
    expr="category == 'tech'",
    output_fields=["title", "url"],
)

# 混合搜索
from pymilvus import AnnSearchRequest, RRFRanker
dense_req = AnnSearchRequest(data=[dense_vec], anns_field="dense_emb",
                              param={"metric_type": "L2", "params": {"ef": 64}}, limit=10)
sparse_req = AnnSearchRequest(data=[sparse_vec], anns_field="sparse_emb",
                               param={"metric_type": "IP"}, limit=10)
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(),
    limit=10,
)
```

### Schema 定义
```python
from pymilvus import CollectionSchema, FieldSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
]
schema = CollectionSchema(fields, description="知识库")
```

## 四、与 FAISS / Pinecone 对比

| 维度 | Milvus | FAISS | Pinecone | Weaviate | Qdrant |
|------|--------|-------|----------|----------|--------|
| 类型 | 分布式向量数据库 | 向量检索库 | 托管服务 | 分布式向量数据库 | 分布式向量数据库 |
| 部署 | Docker/K8s/云 | 嵌入式/C++ | SaaS | Docker/K8s/云 | Docker/K8s/云 |
| 向量类型 | 稠密+稀疏 | 仅稠密 | 仅稠密 | 稠密+稀疏 | 稠密+稀疏 |
| 标量过滤 | ✅ 原生 | ❌ 需后处理 | ✅ | ✅ | ✅ |
| 混合搜索 | ✅ | ❌ | ✅ | ✅ | ✅ |
| 实时写入 | ✅ | ❌ | ✅ | ✅ | ✅ |
| 增量更新 | ✅ | ❌ | ✅ | ✅ | ✅ |
| 去重 | ✅ | ❌ | ✅ | ✅ | ✅ |
| 社区活跃度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 学习曲线 | 中 | 低 | 低 | 中 | 中 |
| 适合场景 | 生产环境 | 实验/嵌入 | 快速上线 | 生产环境 | 生产环境 |

## 五、性能调优建议

### 索引选择
- < 10万向量 → FLAT（简单粗暴最可靠）
- 10万~500万 → IVF_FLAT（通用首选）或 HNSW（追求精度）
- > 500万 → HNSW / DISKANN
- 内存紧张 → IVF_SQ8（节省75%内存）

### 搜索参数
- IVF: nprobe ≈ sqrt(nlist) ~ nlist/4
- HNSW: ef > k（返回数量），通常 ef = 50~500
- 精度优先 → 增大 nprobe/ef
- 速度优先 → 减小 nprobe/ef

### 批量操作
- 插入：每批 1000~10000 条，不要一次插太多
- 搜索：批量查询（一次搜多个向量）比循环快
- Flush：不要频繁 flush，依赖后台刷盘

## 六、常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| Connection refused | Milvus 没启动 | docker ps 检查 |
| Dimension mismatch | 向量维度不匹配 | 检查 embedding 模型输出维度 |
| Collection not loaded | 搜索前未 load | collection.load() |
| Index not found | 索引创建失败 | 检查 index_params 格式 |
| Rate limit exceeded | API 调用太频繁 | 降低频率或使用批量接口 |
| Out of memory | 数据量 > 可用内存 | 用 DISKANN 或增加内存 |
