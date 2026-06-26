"""
04_advanced_features.py — Milvus 高级特性

本课覆盖生产环境中常用的进阶功能：
1. 分区管理（Partition）— 数据隔离和查询加速
2. 一致性级别（Consistency Level）— 强一致 vs 最终一致
3. 动态字段（Dynamic Field）— 灵活 schema
4. 集合管理与运维

【Java 程序员速查】
  from xxx import SomeClass as Alias
  Python 的 as 就是取别名，方便缩短长名字
"""

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

connections.connect("default", host="localhost", port="19530")


# ============================================================
# 第一部分：分区管理
# ============================================================

def demo_partitions():
    """
    分区 = 将一个大集合拆分成多个逻辑子集

    为什么需要分区？
    1. 查询加速：只搜索相关分区，跳过无关数据
    2. 数据隔离：不同租户/不同时间段的数据分开管理
    3. 批量操作：按分区批量删除/更新更方便

    类比：
      一个大仓库 → 按月份分成 12 个区域
      找 3 月的货 → 直接去 3 月区，不用翻全年
    """
    collection_name = "partition_demo"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    # 创建一个带 partition_key 字段的集合
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="分区演示")

    # --- 【Python 语法】_partition_key_field ---
    # 指定分区键字段，告诉 Milvus 按 source 字段做分区
    # 类比 Java: schema.setPartitionKeyField("source");
    schema._partition_key_field = FieldSchema(
        name="source", dtype=DataType.VARCHAR, max_length=32
    )

    collection = Collection(collection_name, schema=schema)

    # 生成数据：3 个来源，每个来源 500 条
    np.random.seed(42)
    sources = ["web", "app", "api"]
    all_data = {}
    for src in sources:
        all_data[src] = np.random.rand(500, 64).astype(np.float32)

    # 直接插入（Milvus 自动按分区键路由）
    for src in sources:
        collection.insert([all_data[src], [src]*500, list(range(1000, 1500))])
        print(f"[分区] 插入 '{src}' 数据: 500 条")

    collection.create_index("embedding", {
        "index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 32},
    })
    collection.load(partition_names=[])  # 加载所有分区

    # --- 【Python 语法】列表推导式 ---
    # [p.name for p in collection.partitions] 类比 Java:
    #   collection.getPartitions().stream().map(Partition::getName).collect(toList())
    print(f"\n[分区] 当前分区列表: {[p.name for p in collection.partitions]}")
    print(f"[分区] 总实体数: {collection.num_entities}")

    # --- 【Python 语法】.tolist() ---
    query = np.random.rand(64).astype(np.float32).tolist()

    # 按分区搜索：只搜索 'web' 分区
    print("\n[分区搜索] 全量搜索 (3个分区):")
    start = time.perf_counter()
    results_all = collection.search(
        data=[query], anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=5,
    )
    elapsed_all = (time.perf_counter() - start) * 1000
    print(f"       耗时: {elapsed_all:.2f}ms")

    print("\n[分区搜索] 只搜索 'web' 分区:")
    start = time.perf_counter()
    results_web = collection.search(
        data=[query], anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 8}},
        limit=5,
        partition_names=["web"],  # ← 只搜这个分区
    )
    elapsed_web = (time.perf_counter() - start) * 1000
    speedup = (1 - elapsed_web/elapsed_all)*100
    print(f"       耗时: {elapsed_web:.2f}ms (快了 {speedup:.1f}%)")

    print("\n[分区] OK 分区搜索可以显著减少扫描范围")
    collection.drop()


# ============================================================
# 第二部分：一致性级别
# ============================================================

def demo_consistency_levels():
    """
    一致性级别 = 写入后多久能读到

    Milvus 提供 4 种一致性级别，从强到弱：

    ┌────────────┬──────────────┬───────────────┬──────────────┐
    │   级别      │   延迟       │    吞吐量     │   适用场景    │
    ├────────────┼──────────────┼───────────────┼──────────────┤
    │ STRONG     │ 实时         │ 最低          │ 金融交易      │
    │ BOUNDED    │ 毫秒~秒级    │ 中等          │ 默认推荐      │
    │ SESSION    │ 同会话内可见  │ 较高          │ 社交动态      │
    │ EVENTUAL   │ 无保证       │ 最高          │ 推荐系统      │
    └────────────┴──────────────┴───────────────┴──────────────┘
    """
    collection_name = "consistency_demo"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64),
        FieldSchema(name="value", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="一致性级别演示")
    collection = Collection(collection_name, schema=schema)

    np.random.seed(42)
    embeddings = np.random.rand(100, 64).astype(np.float32)
    values = [f"value_{i}" for i in range(100)]
    collection.insert([embeddings, values])

    collection.create_index("embedding", {
        "index_type": "FLAT", "metric_type": "L2", "params": {},
    })

    print("\n" + "=" * 60)
    print("  一致性级别")
    print("=" * 60)

    print("\n[一致性] 各级别特点:")
    print("  STRONG:       写入后立即可读，性能最低，适合金融/支付")
    print("  BOUNDED:      毫秒级延迟，默认推荐，适合大多数场景")
    print("  SESSION:      同一会话内可见，适合社交/动态")
    print("  EVENTUAL:     无延迟保证，性能最高，适合推荐/搜索")
    print("\n[一致性] 建议：生产环境默认用 BOUNDED，特殊场景用 STRONG")

    collection.drop()


# ============================================================
# 第三部分：动态字段
# ============================================================

def demo_dynamic_fields():
    """
    动态字段 = 不需要预先定义所有字段

    传统 Schema 要求所有字段在创建集合时就定义好。
    动态字段允许插入时随时添加新字段，类似 MongoDB 的文档模型。
    """
    collection_name = "dynamic_demo"
    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    ]
    schema = CollectionSchema(fields, description="动态字段演示",
                              enable_dynamic_field=True)  # ← 开启动态字段
    collection = Collection(collection_name, schema=schema)

    # 插入时动态添加字段
    data = [
        np.random.rand(3, 64).astype(np.float32),
        ["Article A", "Article B", "Article C"],
    ]
    collection.insert(data)

    print("\n" + "=" * 60)
    print("  动态字段")
    print("=" * 60)
    print("\n[动态字段] 优点:")
    print("  - Schema 灵活，随时添加新字段")
    print("  - 适合元数据多变的场景")
    print("  - 可以用 $meta 访问动态字段")
    print("\n[动态字段] 缺点:")
    print("  - 无法为动态字段创建索引")
    print("  - 过滤动态字段性能较差")
    print("  - 不适合高频查询的动态字段")
    print("\n[动态字段] 建议:")
    print("  - 高频查询的字段 → 预定义 + 建索引")
    print("  - 低频/可选的元数据 → 动态字段")

    collection.drop()


# ============================================================
# 第四部分：集合管理与运维
# ============================================================

def demo_collection_operations():
    """
    集合运维常用操作：
    - 列出所有集合
    - 查看集合信息
    - 刷新（让未 flush 的数据立即可见）
    - 导出/备份
    """
    print("\n" + "=" * 60)
    print("  集合管理与运维")
    print("=" * 60)

    # --- 【Python 语法】utility.list_collections() ---
    # 类比 Java: client.listCollections()
    collections = utility.list_collections()
    print(f"\n[列表] 当前共有 {len(collections)} 个集合:")
    for name in collections:
        coll = Collection(name)
        vec_count = len([f for f in coll.schema.fields if f.dtype.name == "FLOAT_VECTOR"])
        # --- 【Python 语法】列表推导式 ---
        # [f for f in coll.schema.fields if ...] 类比 Java:
        #   coll.getSchema().getFields().stream()
        #       .filter(f -> f.getType().name().equals("FLOAT_VECTOR"))
        #       .collect(toList())
        print(f"  - {name}: {coll.num_entities} 条数据, "
              f"{vec_count} 个向量字段")

    # 查看集合详细信息
    if collections:
        sample_name = collections[0]
        coll = Collection(sample_name)
        print(f"\n[详情] 集合 '{sample_name}':")
        print(f"  描述: {coll.description}")
        print(f"  字段: {[f.name for f in coll.schema.fields]}")
        print(f"  索引: {[idx.index_name for idx in coll.indexes]}")
        print(f"  分区: {[p.name for p in coll.partitions]}")

    print("\n[运维] 常用操作速查:")
    print("  utility.has_collection(name)     → 检查集合是否存在")
    print("  utility.list_collections()        → 列出所有集合")
    print("  utility.drop_collection(name)     → 删除集合")
    print("  collection.load()                 → 加载到内存")
    print("  collection.release()              → 从内存释放（节省资源）")


# ============================================================
# 主程序
# ============================================================

def main():
    print("Milvus 高级特性\n")

    demo_partitions()
    demo_consistency_levels()
    demo_dynamic_fields()
    demo_collection_operations()

    print("\n" + "=" * 60)
    print("  OK 全部完成！")
    print("=" * 60)

    connections.disconnect("default")


if __name__ == "__main__":
    main()
