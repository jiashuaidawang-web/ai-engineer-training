"""
02_vector_search.py — 向量搜索深度解析

本课深入理解向量搜索的核心要素：
1. 不同的距离度量方式及其数学含义
2. 各种索引类型的原理和适用场景
3. 搜索参数调优策略
4. 多向量批量搜索

【Java 程序员速查】
  from xxx import yyy, zzz  类比 Java: import xxx.yyy; import xxx.zzz;
"""

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time

connections.connect("default", host="localhost", port="19530")

COLLECTION_NAME = "vector_search_demo"

# --- 【Python 语法】utility.has_collection() + Collection.drop() ---
# 清理旧的集合，类比 Java: if (client.hasCollection("xxx")) client.dropCollection("xxx");
if utility.has_collection(COLLECTION_NAME):
    Collection(COLLECTION_NAME).drop()

# --- 【Python 语法】列表推导式 ---
# 类比 Java: List<FieldSchema> fields = new ArrayList<>(); fields.add(...);
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="score", dtype=DataType.INT64),
]
schema = CollectionSchema(fields, description="向量搜索深度解析")
collection = Collection(COLLECTION_NAME, schema=schema)

# --- 【Python 语法】np.random.seed() + 列表推导式 ---
np.random.seed(42)  # 固定随机种子（类比 Java: new Random(42)）
n = 5000

# --- 【Python 语法】列表推导式 ---
# [f"label_{i % 10}" for i in range(n)] 类比 Java:
#   for (int i = 0; i < n; i++) labels.add("label_" + (i % 10));
# range(n) 生成 0, 1, 2, ..., n-1（类似 Java 的 IntStream.range(0, n)）
embeddings = np.random.rand(n, 128).astype(np.float32)
labels = [f"label_{i % 10}" for i in range(n)]
scores = [i % 100 for i in range(n)]

print(f"[插入] 正在插入 {n} 条数据...")
collection.insert([embeddings, labels, scores])
print(f"[插入] OK 完成，共 {collection.num_entities} 条\n")


# ============================================================
# 第一部分：三种距离度量的区别
# ============================================================

def demo_distance_metrics():
    """
    演示 L2、Cosine、IP 三种距离度量的差异

    这是向量搜索中最核心的概念之一 — 决定了「相似」的定义。
    """
    metrics = ["L2", "COSINE", "IP"]

    print("=" * 60)
    print("  三种距离度量对比")
    print("=" * 60)

    for metric in metrics:
        # --- 【Python 语法】collection.release() + drop_index() ---
        # pymilvus 3.0 要求：先 release 才能 drop_index
        collection.release()
        collection.drop_index()

        # 创建对应度量的索引（FLAT = 暴力搜索，保证结果绝对准确）
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "FLAT", "metric_type": metric, "params": {}},
        )
        collection.load()

        k = 5
        # --- 【Python 语法】.tolist() ---
        # np.random.rand(128) 返回 numpy array，pymilvus 3.0 要求传入 Python list
        # 类比 Java: 从 float[] 转为 List<Float>
        query = np.random.rand(128).astype(np.float32).tolist()

        results = collection.search(
            data=[query],
            anns_field="embedding",
            param={"metric_type": metric, "params": {}},
            limit=k,
            output_fields=["label", "score"],
        )

        # --- 【Python 语法】列表推导式 ---
        # distances = [hit.distance for hit in results[0]]
        # 类比 Java: List<Double> distances = results.get(0).stream()
        #                                      .map(hit -> hit.getDistance())
        #                                      .collect(Collectors.toList());
        distances = [hit.distance for hit in results[0]]
        avg_dist = sum(distances) / len(distances)  # Python 用内置 sum()/len()

        print(f"\n[{metric}] Top-{k} 结果:")
        print(f"  平均距离: {avg_dist:.4f}")
        print(f"  距离范围: [{min(distances):.4f}, {max(distances):.4f}]")
        print(f"  Top3: ")
        for i, hit in enumerate(results[0][:3]):
            # --- 【Python 语法】切片 [:3] ---
            # results[0][:3] 取前 3 个元素（类比 Java: subList(0, 3)）
            print(f"    #{i+1} id={hit.id} dist={hit.distance:.4f} label={hit.fields['label']}")


# ============================================================
# 第二部分：索引类型深度解析
# ============================================================

def demo_index_types():
    """
    对比不同索引类型的性能特征
    """
    # --- 【Python 语法】元组(tuple) ---
    # 元组是不可变的列表，类比 Java 的 record / 自定义 DTO
    index_configs = [
        # (索引名, index_type, metric_type, params, 描述)
        ("flat",        "FLAT",         "L2",         {},          "暴力搜索，绝对精确，适合小数据"),
        ("ivf_flat",    "IVF_FLAT",     "L2",         {"nlist": 128},  "经典聚类索引，通用首选"),
        ("ivf_sq8",     "IVF_SQ8",      "L2",         {"nlist": 128},  "IVF + 8位标量量化，节省75%内存"),
        ("hnsw",        "HNSW",         "L2",         {"M": 16, "efConstruction": 200},
                        "图索引，搜索质量最高"),
    ]

    print("\n" + "=" * 60)
    print("  索引类型对比")
    print("=" * 60)

    # --- 【Python 语法】多重解包 ---
    # for idx_name, idx_type, metric, params, desc in index_configs
    # 类比 Java: for (Config cfg : configs) { String idxName = cfg.getIdxName(); ... }
    for idx_name, idx_type, metric, params, desc in index_configs:
        try:
            # 先 release 并删除旧索引
            collection.release()
            collection.drop_index()
            collection.create_index(
                field_name="embedding",
                index_params={"index_type": idx_type, "metric_type": metric, "params": params},
                index_name=idx_name,
            )
            print(f"\n[OK] {idx_name.upper()} — {desc}")
            print(f"    参数: {params}")
        except Exception as e:
            # --- 【Python 语法】异常捕获 ---
            # 类比 Java: catch (Exception e)
            print(f"\n[X] {idx_name} 创建失败: {e}")

    # 加载最后一个成功的索引并测试
    last_idx = index_configs[-1][0]  # [-1] 是 Python 的负索引（取最后一个元素）
    # 类比 Java: indexConfigs.get(indexConfigs.size() - 1)[0]

    try:
        collection.load()

        # 生成查询向量 — 转为 Python list（pymilvus 3.0 要求）
        query = np.random.rand(128).astype(np.float32).tolist()

        # HNSW 需要 ef 参数，IVF 需要 nprobe 参数
        # --- 【Python 语法】三元表达式 ---
        # (A if condition else B) 类比 Java: condition ? A : B
        if last_idx == "hnsw":
            search_param = {"metric_type": "L2", "params": {"ef": 64}}
        elif last_idx.startswith("ivf"):
            search_param = {"metric_type": "L2", "params": {"nprobe": 16}}
        else:
            search_param = {"metric_type": "L2", "params": {}}

        start = time.perf_counter()  # 高精度计时，类比 Java: System.nanoTime()
        results = collection.search(
            data=[query],
            anns_field="embedding",
            param=search_param,
            limit=10,
            output_fields=["label"],
        )
        elapsed = time.perf_counter() - start  # 计算耗时

        print(f"\n[搜索] 耗时: {elapsed*1000:.2f}ms")
        print(f"       Top-3: ", end="")
        for i, hit in enumerate(results[0][:3]):
            # --- 【Python 语法】print(end=" ") ---
            # 不换行输出，类比 Java: System.out.print(...)
            print(f"#{i+1}(id={hit.id}, dist={hit.distance:.4f}) ", end="")
        print()  # 换行

    except Exception as e:
        print(f"\n[搜索] 失败: {e}")


# ============================================================
# 第三部分：搜索参数调优 — ef 和 nprobe
# ============================================================

def demo_search_parameters():
    """
    演示搜索参数的调优

    每个索引类型有不同的搜索参数，调优这些参数可以在
    速度和质量之间找到最佳平衡点。

    IVF 系列: nprobe（考察的簇数量）
      nprobe=1   → 最快，但可能漏掉最近邻
      nprobe=nlist → 等于暴力搜索，最慢但最准
      推荐: sqrt(nlist) ~ nlist/4

    HNSW: ef（搜索时考虑的节点数量）
      ef=1     → 最快，质量最低
      ef=collection_size → 等于暴力搜索
      推荐: 略大于返回的 k 值（如 k=10 时 ef=50~100）
    """
    print("\n" + "=" * 60)
    print("  搜索参数调优 — HNSW 的 ef 参数")
    print("=" * 60)

    # 确保 HNSW 索引存在
    # pymilvus 3.0: 先 release 旧索引，再 load，再检查
    collection.release()
    collection.drop_index()
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "HNSW", "metric_type": "L2",
                      "params": {"M": 16, "efConstruction": 200}},
        index_name="hnsw_ef_demo",
    )
    collection.load()

    query = np.random.rand(128).astype(np.float32).tolist()
    k = 10

    # --- 【Python 语法】print 格式化表头 ---
    # f"{'ef':<8}" — 左对齐占 8 格
    # 类比 Java: String.format("%-8s", "ef")
    print(f"\n{'ef':<8} {'耗时(ms)':<12} {'Top-1距离':<14} {'Top-10距离范围'}")
    print("-" * 60)

    # --- 【Python 语法】for 循环遍历列表 ---
    # 类比 Java: for (int ef : efValues) { ... }
    for ef in [10, 50, 100, 500]:
        start = time.perf_counter()
        results = collection.search(
            data=[query],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": ef}},
            limit=k,
        )
        elapsed = (time.perf_counter() - start) * 1000

        # --- 【Python 语法】列表推导式 ---
        # [hit.distance for hit in results[0]] 类比 Java:
        #   results.get(0).stream().mapToDouble(Hit::getDistance).toArray()
        distances = [hit.distance for hit in results[0]]
        dist_range = f"[{min(distances):.4f}, {max(distances):.4f}]"

        print(f"{ef:<8} {elapsed:<12.2f} {distances[0]:<14.4f} {dist_range}")


# ============================================================
# 第四部分：批量搜索（一次搜多个向量）
# ============================================================

def demo_batch_search():
    """
    批量搜索：一次查询多个向量

    适用场景：
    - 用户同时上传多张图片找相似商品
    - 多语言翻译后同时搜索
    - A/B 测试不同 embedding 的效果
    """
    print("\n" + "=" * 60)
    print("  批量搜索")
    print("=" * 60)

    # --- 【Python 语法】列表推导式 ---
    # [np.random.rand(128)...tolist() for _ in range(3)]
    # 生成 3 个查询向量，每个是 128 维的 Python list
    # 类比 Java:
    #   List<float[]> batchQueries = new ArrayList<>();
    #   for (int i = 0; i < 3; i++) batchQueries.add(new float[]{...});
    batch_queries = [np.random.rand(128).astype(np.float32).tolist() for _ in range(3)]

    # 一次搜索 3 个向量，每个返回 Top-3
    results = collection.search(
        data=batch_queries,       # 传入多个查询向量
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        limit=3,
        output_fields=["label"],
    )

    # --- 【Python 语法】嵌套列表遍历 ---
    # results 是一个嵌套列表: [[hit1, hit2, hit3], [hit4, hit5, hit6], ...]
    # 外层列表对应每个查询，内层列表对应该查询的命中结果
    # 类比 Java: List<List<Hit>> results;
    #   for (int i = 0; i < results.size(); i++)
    #     for (Hit hit : results.get(i))
    for i, query_results in enumerate(results):
        print(f"\n[查询向量 #{i+1}] Top-3:")
        for j, hit in enumerate(query_results):
            # --- 【Python 语法】dict.get() 安全取值 ---
            # hit.fields.get('label', 'N/A') 类比 Java:
            #   hit.getFields().getOrDefault("label", "N/A")
            print(f"  #{j+1} id={hit.id} dist={hit.distance:.4f} label={hit.fields.get('label', 'N/A')}")


# ============================================================
# 主程序
# ============================================================

def main():
    print("Milvus 向量搜索深度解析")
    print("运行前请确保 Milvus 服务已启动\n")

    demo_distance_metrics()
    demo_index_types()
    demo_search_parameters()
    demo_batch_search()

    print("\n" + "=" * 60)
    print("  OK 全部完成！")
    print("=" * 60)

    collection.drop()
    connections.disconnect("default")


if __name__ == "__main__":
    main()
