"""
03_filter_hybrid_search.py — 标量过滤 + 混合搜索

本课学习 Milvus 最强大的能力之一：
在向量搜索的同时，结合标量字段的过滤条件。

核心概念：
1. 标量过滤搜索（Scalar Filter Search）
2. 稀疏向量搜索（Sparse Vector）— BM25 关键词匹配
3. 稠密+稀疏混合搜索（Dense-Sparse Hybrid Search）— 语义+关键词
4. 加权融合（RRF / Weighted）

【Java 程序员速查】
  from xxx import yyy, zzz as aaa
  类比 Java: import xxx.yyy; import xxx.zzz aaa; (Java 不支持 as)
  Python 的 as 就是取别名，方便缩短长名字
"""

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility,
    AnnSearchRequest, RRFRanker, WeightedRanker,
)
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

connections.connect("default", host="localhost", port="19530")

COLLECTION_NAME = "hybrid_search_demo"

if utility.has_collection(COLLECTION_NAME):
    Collection(COLLECTION_NAME).drop()


# ============================================================
# 第一部分：标量过滤搜索
# ============================================================

def demo_scalar_filter():
    """
    标量过滤 = 在向量搜索中加上 WHERE 条件

    这是 Milvus 相比 FAISS 的巨大优势：
    - FAISS: 只能做纯向量搜索，过滤需要自己后处理
    - Milvus: 在服务端直接过滤，利用标量索引加速
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="price", dtype=DataType.INT64),
        FieldSchema(name="is_active", dtype=DataType.BOOL),
        FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR,
                    max_capacity=10, max_length=32),
    ]
    schema = CollectionSchema(fields, description="标量过滤演示")
    collection = Collection(COLLECTION_NAME, schema=schema)

    np.random.seed(42)
    n = 1000
    embeddings = np.random.rand(n, 128).astype(np.float32)
    categories = ["electronics", "clothing", "food", "books"] * 250
    prices = [random.randint(10, 10000) for _ in range(n)]
    is_active = [i % 3 != 0 for i in range(n)]
    tags_list = [
        random.sample(["sale", "new", "hot", "vip", "free_shipping", "limited"], k=2)
        for _ in range(n)
    ]

    collection.insert([embeddings, categories, prices, is_active, tags_list])
    collection.create_index("embedding", {
        "index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 64}
    })
    collection.load()

    # --- 【Python 语法】.tolist() ---
    # pymilvus 3.0 要求查询向量是 Python list，不是 numpy array
    query = np.random.rand(128).astype(np.float32).tolist()

    print("=" * 60)
    print("  标量过滤搜索")
    print("=" * 60)

    # --- 【Python 语法】带 expr 参数的搜索 ---
    # expr 就是 SQL 的 WHERE 子句
    # 类比 Java: searchParam.setExpr("category == 'electronics'");

    # 过滤表达式 1: 单条件
    expr1 = "category == 'electronics'"
    results1 = collection.search(
        data=[query], anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=5, expr=expr1, output_fields=["category", "price"],
    )
    print(f"\n[过滤1] {expr1}")
    print(f"{'ID':<6} {'距离':<12} {'类别':<15} {'价格'}")
    for hit in results1[0]:
        print(f"{hit.id:<6} {hit.distance:<12.4f} {hit.fields['category']:<15} {hit.fields['price']}")

    # 过滤表达式 2: 范围查询（用 in 替代，Milvus 2.4 对 numeric range 支持有限）
    expr2 = "score > 50"
    results2 = collection.search(
        data=[query], anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=5, expr=expr2, output_fields=["category", "price"],
    )
    print(f"\n[过滤2] {expr2}")
    print(f"{'ID':<6} {'距离':<12} {'类别':<15} {'价格'}")
    for hit in results2[0]:
        print(f"{hit.id:<6} {hit.distance:<12.4f} {hit.fields['category']:<15} {hit.fields['price']}")

    # 过滤表达式 3: in 操作
    expr3 = "category in ['electronics', 'books']"
    results3 = collection.search(
        data=[query], anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=5, expr=expr3, output_fields=["category", "price"],
    )
    print(f"\n[过滤3] {expr3}")
    print(f"{'ID':<6} {'距离':<12} {'类别':<15} {'价格'}")
    for hit in results3[0]:
        print(f"{hit.id:<6} {hit.distance:<12.4f} {hit.fields['category']:<15} {hit.fields['price']}")

    # 过滤表达式 4: 数组字段 contains
    expr4 = "tags contains 'sale'"
    results4 = collection.search(
        data=[query], anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=5, expr=expr4, output_fields=["tags"],
    )
    print(f"\n[过滤4] {expr4}")
    print(f"{'ID':<6} {'距离':<12} {'Tags'}")
    for hit in results4[0]:
        print(f"{hit.id:<6} {hit.distance:<12.4f} {hit.fields['tags']}")

    collection.drop()


# ============================================================
# 第二部分：混合搜索（Dense + Sparse）
# ============================================================

def demo_hybrid_search():
    """
    混合搜索 = 向量语义匹配 + 关键词精确匹配

    为什么需要混合搜索？
    ┌─────────────┬──────────────────────┬──────────────────────┐
    │             │    向量搜索(Dense)    │   关键词搜索(Sparse)  │
    ├─────────────┼──────────────────────┼──────────────────────┤
    │ 优势        │ 语义理解，不怕措辞不同  │ 精确匹配，不会幻觉      │
    │ 劣势        │ 可能找不到精确关键词     │ 无法理解语义           │
    │ 例子        │ "苹果" 能找到 "iPhone"  │ "Milvus" 精确找到文档  │
    └─────────────┴──────────────────────┴──────────────────────┘

    混合搜索 = Dense（语义） + Sparse（关键词） → 互补

    融合方式：
    1. Weighted: 加权求和，score = alpha*dense + (1-alpha)*sparse
    2. RRF (Reciprocal Rank Fusion): 按排名倒数加权融合，无需调权重
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
    ]
    schema = CollectionSchema(
        fields,
        description="混合搜索演示",
        vectors_meta="fields=dense_embedding,sparse_embedding",
    )
    collection = Collection(f"{COLLECTION_NAME}_hybrid", schema=schema)

    np.random.seed(42)
    random.seed(42)
    n = 500

    dense_embeddings = np.random.rand(n, 128).astype(np.float32)

    # --- 【Python 语法】字典推导式 ---
    # {str(t): round(random.uniform(0.1, 1.0), 4) for t in terms}
    # 类比 Java: Map<Integer, Double> weights = new HashMap<>();
    #            for (int t : terms) weights.put(t, round(...));
    sparse_embeddings = []
    for i in range(n):
        num_terms = random.randint(3, 8)
        terms = random.sample(range(1000), num_terms)
        weights = {str(t): round(random.uniform(0.1, 1.0), 4) for t in terms}
        sparse_embeddings.append(weights)

    titles = [f"Document {i} - {random.choice(['AI', 'ML', 'Deep Learning', 'NLP', 'CV'])}" for i in range(n)]
    categories = ["tech", "science", "business", "health"] * 125

    collection.insert([dense_embeddings, sparse_embeddings, titles, categories])

    collection.create_index("dense_embedding", {
        "index_type": "HNSW", "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 200},
    })
    collection.create_index("sparse_embedding", {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IR_PROBABILISTIC",
        "params": {"threshold": 0.2, "build_mode": 1},
    })
    collection.load()

    print("\n" + "=" * 60)
    print("  混合搜索（Dense + Sparse）")
    print("=" * 60)

    # --- 【Python 语法】AnnSearchRequest ---
    # 类比 Java:
    #   SearchRequest denseReq = SearchRequest.builder()
    #       .annField("dense_embedding").data(query_dense)
    #       .param(SearchParam.builder().metricType(L2).ef(64).build())
    #       .limit(10).build();
    query_dense = np.random.rand(128).astype(np.float32).tolist()
    query_sparse = {str(i): round(random.uniform(0.1, 1.0), 4) for i in random.sample(range(1000), 5)}

    dense_req = AnnSearchRequest(
        data=[query_dense],
        anns_field="dense_embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=10,
    )
    sparse_req = AnnSearchRequest(
        data=[query_sparse],
        anns_field="sparse_embedding",
        param={"metric_type": "IR_PROBABILISTIC", "params": {"threshold": 0.2}},
        limit=10,
    )

    # 方式1: Weighted 融合
    ranker = WeightedRanker(0.7, 0.3)  # dense 占 70%，sparse 占 30%
    results_weighted = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        ranker=ranker,
        limit=10,
        output_fields=["title", "category"],
    )

    print("\n[Weighted 融合] dense=0.7, sparse=0.3")
    print(f"{'排名':<4} {'ID':<8} {'标题':<50} {'类别'}")
    print("-" * 75)
    for i, hits in enumerate(results_weighted):
        hit = hits[0]
        title = hit.fields.get("title", "")[:48]  # 截断显示
        print(f"{i+1:<4} {hit.id:<8} {title:<50} {hit.fields.get('category', '')}")

    # 方式2: RRF 融合（推荐，无需调权重）
    ranker_rrf = RRFRanker()
    results_rrf = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        ranker=ranker_rrf,
        limit=10,
        output_fields=["title", "category"],
    )

    print("\n[RRF 融合] 倒数排名融合（自动平衡，无需调权重）")
    print(f"{'排名':<4} {'ID':<8} {'标题':<50} {'类别'}")
    print("-" * 75)
    for i, hits in enumerate(results_rrf):
        hit = hits[0]
        title = hit.fields.get("title", "")[:48]
        print(f"{i+1:<4} {hit.id:<8} {title:<50} {hit.fields.get('category', '')}")

    collection.drop()


# ============================================================
# 主程序
# ============================================================

def main():
    print("Milvus 标量过滤与混合搜索\n")

    print(">>> 第一部分：标量过滤搜索")
    demo_scalar_filter()

    print("\n>>> 第二部分：混合搜索")
    demo_hybrid_search()

    print("\n" + "=" * 60)
    print("  OK 全部完成！")
    print("=" * 60)

    connections.disconnect("default")


if __name__ == "__main__":
    main()
