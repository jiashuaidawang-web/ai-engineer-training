"""
01_milvus_primer.py — Milvus 入门：连接、创建集合、插入数据、简单搜索

本文件是 Milvus 学习的第一课，覆盖最基础的 CRUD 操作。
每一步都配有详细注释，解释「做了什么」和「为什么要这样做」。

【Java 程序员速查】
  Python 的 .py 文件 ≈ Java 的 .java 文件（一个文件一个类，但可以没有类）
  Python 没有 public/private，缩进代替了 {} 和大括号
  运行方式: python 01_milvus_primer.py  （相当于 java 编译+运行一步到位）

运行前请确保 Milvus 服务正在运行：
  docker ps | grep milvus
  或 milvus run standalone
"""

# ============================================================
# 【Python 语法】import — 导入模块/包
#   类比 Java: import com.xxx.Yyy;
#   Python 的包 = Java 的包(package)，模块 = Java 的类文件(.java)
# ============================================================

from pymilvus import (
    # pymilvus 是 Milvus 的 Python SDK，类比 Java 的 milvus-client.jar
    # 从 jar 包中 import 类，用法和 Java 完全一样：com.milvus.client.Connections -> Connections
    connections,        # 连接管理器 —— 类比: MilvusGrpcClient / MilvusServiceClient
    Collection,         # 集合操作类 —— 类比: 一个 DAO 类，封装 CRUD
    CollectionSchema,   # 集合结构定义 —— 类比: TableSchema / DDL
    FieldSchema,        # 字段定义 —— 类比: ColumnDefinition
    DataType,           # 数据类型枚举 —— 类比: enum DataType { INT64, FLOAT_VECTOR, ... }
    utility,            # 工具模块 —— 类比: 一堆静态方法的 Utility 类
)
import numpy as np    # NumPy 库 — 科学计算，类比 Java 的 Apache Commons Math + 自定义数组工具
                     # np 是约定俗成的缩写（就像 Java 里常用 log 代替 logger）

import warnings       # Python 内置的警告控制模块
                     # 类比: 日志框架中的 setLevel(WARN) 来屏蔽不需要的输出

# ============================================================
# 【Python 语法】warnings.filterwarnings("ignore")
#   屏蔽所有 DeprecationWarning（弃用警告）
#   pymilvus 3.0 对旧 API 标记了「即将废弃」的警告，学习阶段先关掉
#   类比: slf4j 中 @SuppressWarnings("deprecation")
# ============================================================

warnings.filterwarnings("ignore")

# ============================================================
# 【Python 语法】常量命名
#   Python 惯例：全局常量用 UPPER_CASE（虽然 Python 没有 final 关键字）
#   类比 Java: private static final String HOST = "localhost";
# ============================================================

HOST = "localhost"     # Milvus 服务器地址
PORT = "19530"         # Milvus 默认端口

# ============================================================
# 【Python 语法】def — 定义函数
#   类比 Java: public void connectMilvus() { ... }
#   区别：
#     1. Python 不需要写返回值类型（动态类型）
#     2. 不需要写访问修饰符（public/private，Python 靠约定）
#     3. 用冒号 : 代替大括号 {}，用缩进表示代码块
# ============================================================


def connect_milvus():
    """
    【Python 语法】docstring — 用三重引号包裹的字符串
      这是函数的「文档字符串」，Java 类比: Javadoc /** ... */
      可以用 help(connect_milvus) 查看
    """
    # 建立连接到 Milvus 服务

    # --- 【Python 语法】try/except ---
    # 类比 Java: try { ... } catch (Exception e) { ... }
    # Python 的 except 不需要指定异常类型时，可以省略括号里的内容
    try:
        # disconnect("default") — 断开名为 "default" 的连接
        # 类比 Java: client.close(); 但 Milvus 支持多连接，用 alias 区分
        connections.disconnect("default")
    except Exception:
        # 如果连接不存在，disconnect 会抛异常，忽略即可
        pass              # pass 是空语句，类比 Java: {} 空代码块

    # --- 【Python 语法】f-string (格式化字符串) ---
    # f"[连接] 正在连接到 Milvus: {HOST}:{PORT}"
    # 类比 Java: String.format("[连接] 正在连接到 Milvus: %s:%s", HOST, PORT)
    # 或者: "[连接] 正在连接到 Milvus: " + HOST + ":" + PORT
    print(f"[连接] 正在连接到 Milvus: {HOST}:{PORT}")

    # connect(alias="default", host=HOST, port=PORT)
    # 类比 Java: connections.connect("default", "localhost", "19530");
    # alias="default" 是命名参数（Java 没有这个语法，需要重载构造函数或 Builder）
    connections.connect(alias="default", host=HOST, port=PORT)
    print("[连接] OK 连接成功\n")


# ============================================================
# 【概念】Collection（集合）— 向量数据库中的「表」
#
# 一个 Collection 包含：
#   1. Schema（数据结构定义）：有哪些字段，各是什么类型
#   2. Data（数据）：实际存储的行记录
#   3. Index（索引）：加速搜索的数据结构
#
# Schema 中的字段类型：
#   - PRIMARY_KEY: 主键，唯一标识每条数据（通常是 int64 或 varchar）
#   - VECTOR: 向量字段，存储 embedding 向量
#   - scalar fields: 普通字段（int64, varchar, bool 等），用于过滤和返回
# ============================================================

COLLECTION_NAME = "hello_milvus"  # 集合名称（类似数据库中的表名）


def setup_collection():
    """
    定义并创建集合

    每条数据包含三个字段：
    1. id:       主键（int64），自增
    2. embedding: 向量（float vector），维度 128
    3. category: 分类标签（varchar），用于标量过滤

    类比 SQL:
      CREATE TABLE hello_milvus (
          id        BIGINT PRIMARY KEY AUTO_INCREMENT,
          embedding FLOAT[128],          ← 向量字段（特殊）
          category  VARCHAR(64)          ← 标量字段
      );
    """
    # 清理：删除已存在的同名集合（方便重复运行）
    # ⚠️ 生产环境中不要随意删除集合！

    # --- 【Python 语法】utility.has_collection() ---
    # 类比 Java: collectionManager.hasCollection("hello_milvus")
    if utility.has_collection(COLLECTION_NAME):
        print(f"[清理] 删除已有集合 '{COLLECTION_NAME}'")

        # --- 【Python 语法】Collection(COLLECTION_NAME).drop() ---
        # 先构造一个 Collection 对象（拿到表的引用），再调用 drop() 删除
        # 类比 Java: new Collection("hello_milvus").drop();
        Collection(COLLECTION_NAME).drop()

    # --- 【Python 语法】return 的隐式 None ---
    # Python 函数可以不写 return，此时返回 null（叫 None）
    # 类比 Java: void 方法不写 return
    return None


def create_schema():
    """
    定义集合的 Schema（数据结构）

    类比 Java:
      TableSchema schema = TableSchema.newBuilder()
          .addColumn(ColumnDef.newBuilder()...build())
          .build();
    """
    # --- 【Python 语法】列表(list) ---
    # Python 的 list 类比 Java 的 ArrayList，但更灵活
    # 不需要指定泛型: List<FieldSchema> fields = new ArrayList<>();
    # Python 直接写: fields = [...]
    fields = [
        # ── 主键字段 ──
        FieldSchema(
            name="id",              # 字段名 — 类比 Java: .setName("id")
            dtype=DataType.INT64,   # 数据类型 — 类比 Java: .setType(DataType.INT64)
            is_primary=True,        # 是否为主键 — True 是 Python 的关键字（Java 是 true）
            auto_id=True,           # 是否自动生成（True = 数据库自增ID）
        ),
        # ── 向量字段（核心）──
        FieldSchema(
            name="embedding",       # 字段名
            dtype=DataType.FLOAT_VECTOR,  # 向量类型：浮点向量
            dim=128,                # 向量维度！必须与 Embedding 模型的输出维度一致
                                    # 常用维度: text-embedding-ada-002 → 1536
                                    #          bge-large-zh → 1024
                                    #          sentence-transformers → 384
        ),
        # ── 标量字段（辅助）──
        FieldSchema(
            name="category",        # 字段名
            dtype=DataType.VARCHAR, # 字符串类型
            max_length=64,          # 最大长度（VARCHAR 必须指定）
        ),
    ]

    # --- 【Python 语法】对象构造 ---
    # Python 构造对象不需要 new 关键字
    # 类比 Java: CollectionSchema schema = new CollectionSchema(fields, "描述");
    schema = CollectionSchema(
        fields=fields,           # 命名参数 — Java 需要按顺序传参或用 Builder
        description="Hello Milvus 演示集合",
    )
    return schema


def create_collection(schema):
    """
    创建集合

    创建后 Milvus 会：
    1. 分配存储空间
    2. 注册 Schema 元数据
    3. 准备接受数据写入

    类比 Java:
      Collection collection = new Collection(schema);
      collection.load();
    """
    print("[创建集合] 正在创建集合...")

    # --- 【Python 语法】Collection 构造函数 ---
    # 类比 Java: Collection coll = new Collection("hello_milvus", schema);
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        consistency_level="Bounded",  # 数据一致性级别（默认推荐）
    )

    # --- 【Python 语法】点号访问属性 ---
    # collection.num_entities 类比 Java: collection.getNumEntities()
    # Python 的属性可以直接点号访问，不需要 getter/setter
    print(f"[创建集合] OK 集合 '{COLLECTION_NAME}' 已创建，实体数: {collection.num_entities}\n")
    return collection


# ============================================================
# 【概念】数据插入
#
# 插入流程：
#   1. 生成/获取向量（通过 Embedding 模型将文本转为向量）
#   2. 准备标量字段数据（类别标签等）
#   3. 调用 insert() 批量写入
#   4. flush() 将内存数据刷到磁盘（可选，生产环境通常异步刷盘）
# ============================================================

def insert_data(collection):
    """
    向集合中插入模拟数据

    实际项目中，向量来自 Embedding 模型（如 OpenAI text-embedding-ada-002），
    这里用随机向量模拟，但插入方式和真实场景完全一致。
    """
    num_entities = 2000

    # --- 【Python 语法】numpy 生成随机矩阵 ---
    # np.random.rand(2000, 128) 类比 Java:
    #   float[][] embeddings = new float[2000][128];
    #   for(...) for(...) embeddings[i][j] = random.nextFloat();
    # numpy 的优势：底层是 C 实现的，速度比纯 Python 快 10-100 倍
    embeddings = np.random.rand(num_entities, 128).astype(np.float32)
    # astype(np.float32) — 转换数据类型为 32 位浮点
    # 类比 Java: 从 double[][] 转为 float[][]

    # --- 【Python 语法】列表乘法 ---
    # ["tech", "sports", "finance", "entertainment"] * 500
    # 类比 Java: 手动循环 append 500 次
    # 结果: ["tech", "sports", "finance", "entertainment", "tech", "sports", ...]
    categories = ["tech", "sports", "finance", "entertainment"] * (num_entities // 4)
    # // 是整除运算符（类比 Java 的 int / int），/ 会得到小数

    print(f"[插入数据] 正在插入 {num_entities} 条数据...")

    # --- 【Python 语法】insert() 接收列表 ---
    # insert([embeddings, categories]) — 传入一个列表，包含两个字段的数据
    # 类比 Java: collection.insert(new Object[]{embeddings, categories});
    # 或者 Builder 模式: insertBuilder.setEmbeddings(embeddings).setCategories(categories).build()
    insert_result = collection.insert([embeddings, categories])

    # --- 【Python 语法】对象属性访问 ---
    # insert_result.insert_count — 类比 Java: result.getInsertCount()
    # Python 不需要写 getter，直接点号访问
    print(f"[插入数据] OK 插入完成，新增实体数: {insert_result.insert_count}")
    print(f"[插入数据] OK 当前实体数: {collection.num_entities}\n")


# ============================================================
# 【概念】索引 — 向量搜索的「加速器」
#
# 没有索引的向量搜索 = 遍历所有向量逐一计算距离（暴力搜索）
#   100万向量 × 128维 → 每次查询需要 100万×128次乘法加法 → 很慢
#
# 有索引 = 先缩小候选范围，再精确计算 → 快几十到几百倍
#
# 常用索引类型：
# ┌────────────┬──────────────────┬──────────────┬───────────────┐
# │   索引名    │     适用场景      │   构建速度   │    查询速度    │
# ├────────────┼──────────────────┼──────────────┼───────────────┤
# │ FLAT       │ 小规模数据(<10万) │   快         │   慢(暴力搜索) │
# │ IVF_FLAT   │ 中等规模          │   中等       │   快          │
# │ HNSW       │ 大规模(百万+)     │   慢         │   很快        │
# │ DISKANN    │ 超大规模(千万+)   │   慢         │   很快        │
# └────────────┴──────────────────┴──────────────┴───────────────┘
# ============================================================

def create_index(collection):
    """
    为向量字段创建索引

    这里使用 IVF_FLAT 索引：
    - IVF = Inverted File Index（倒排文件索引）
    - 先将向量分成 K 个簇（K 默认为 128）
    - 查询时只计算与查询向量最近的几个簇中的向量
    - 参数 nlist: 簇的数量，越大越精确但越慢

    类比：图书馆的书按主题分类放书架 → 找书时先看主题再翻书架，而不是翻每一本书
    """
    # --- 【Python 语法】字典(dict) ---
    # Python 的 dict 类比 Java 的 HashMap<String, Object>
    # {"index_type": "IVF_FLAT"} 就是 {key: value} 键值对
    # 类比 Java: Map<String, Object> indexParams = new HashMap<>();
    #            indexParams.put("index_type", "IVF_FLAT");
    index_params = {
        "index_type": "IVF_FLAT",  # 索引类型
        "metric_type": "L2",       # 距离度量：L2欧氏距离（越小越相似）
        "params": {"nlist": 128},  # 嵌套字典 — 类比 Java: params.put("nlist", 128);
    }

    print(f"[创建索引] 正在创建索引...")
    # --- 【Python 语法】字典取值 ---
    # index_params['index_type'] 类比 Java: indexParams.get("index_type")
    print(f"           类型: {index_params['index_type']}")
    print(f"           度量: {index_params['metric_type']}")
    print(f"           参数: {index_params['params']}\n")

    # --- 【Python 语法】create_index() ---
    # 类比 Java: collection.createIndex("embedding", indexParams);
    collection.create_index(
        field_name="embedding",   # 对哪个字段建索引
        index_params=index_params, # 索引参数
    )

    # 加载索引到内存（搜索前必须加载）
    # load() 内部已同步等待加载完成，无需额外调用 wait_for_loading_complete
    print("[创建索引] 正在加载索引到内存...")
    collection.load()
    print("[创建索引] OK 索引已加载，可以开始搜索\n")


# ============================================================
# 【概念】向量搜索
#
# 搜索流程：
#   1. 准备查询向量（同 Embedding 模型生成的格式）
#   2. 指定搜索参数：
#      - metric_type: 距离度量方式
#      - limit: 返回最相似的 N 条结果
#      - params: 索引特定的搜索参数（如 IVF 的 nprobe）
#   3. 执行搜索，返回 (结果ID, 距离分数)
# ============================================================

def search(collection):
    """
    执行向量搜索

    搜索的本质：计算查询向量与数据库中每个向量的「距离」，
    返回距离最近的 K 条结果。

    距离度量方式：
    - L2 (Euclidean): 欧氏距离，取值范围 [0, ∞)，越小越相似
    - COSINE: 余弦相似度，取值范围 [-1, 1]，越大越相似（搜索时返回 1-COSINE）
    - IP (Inner Product): 内积，不考虑向量方向时的相似度

    参数 nprobe（IVF 索引专用）：
    - 搜索时考察多少个簇
    - nprobe=1 → 只查最接近的1个簇 → 最快但可能漏掉正确答案
    - nprobe=128 → 查所有簇 → 等于暴力搜索，最慢但最准
    - 通常取 sqrt(nlist) 到 nlist 之间，如 nlist=128 时 nprobe=8~32
    """
    # 生成一个随机查询向量（实际项目中来自用户的文本经 Embedding 模型转换）
    # --- 【Python 语法】np.random.rand(1, 128) ---
    # 生成 1x128 的随机矩阵（注意是二维的，因为 search 要求输入是列表）
    query_vector = np.random.rand(1, 128).astype(np.float32)

    # 搜索参数
    search_params = {
        "metric_type": "L2",      # 与建索引时使用的度量方式一致
        "params": {"nprobe": 10}, # 考察 10 个簇
    }

    k = 5  # 返回最相似的 5 条结果

    print(f"[搜索] 正在搜索 Top-{k} 相似向量...")

    # --- 【Python 语法】collection.search() ---
    # 类比 Java:
    #   SearchParam param = new SearchParam()
    #       .setAnnField("embedding")
    #       .setMetricType(MetricType.L2)
    #       .setNprobe(10)
    #       .setTopK(5);
    #   ResultSet results = collection.search(queryVector, param);
    results = collection.search(
        data=query_vector,           # 查询向量（注意是二维数组）
        anns_field="embedding",      # 在哪个字段中搜索
        param=search_params,         # 搜索参数
        limit=k,                     # 返回结果数量
        output_fields=["category"],  # 除了 ID 和距离，还返回哪些标量字段
    )

    # --- 【Python 语法】results 的结构 ---
    # results 是一个列表，每个元素对应一次查询的结果
    # 因为我们只查了一个向量，所以 results[0] 就是这次查询的所有命中
    # 类比 Java: List<Hit> hits = results.getHits(0);

    # 解析搜索结果
    print("[搜索] OK 搜索结果:")

    # --- 【Python 语法】f-string 格式化 ---
    # f"{'排名':<6}" — 左对齐，占6格
    # 类比 Java: String.format("%-6s", "排名")
    print(f"{'排名':<6} {'ID':<10} {'距离':<12} {'类别'}")
    print("-" * 45)  # "-" * 45 生成 45 个 "-" 字符

    # --- 【Python 语法】enumerate() ---
    # enumerate(results[0]) 返回 (索引, 元素) 的对
    # 类比 Java: for (int i = 0; i < hits.size(); i++) { Hit hit = hits.get(i); }
    for i, hit in enumerate(results[0]):
        # hit 是一个对象，包含:
        #   hit.id      → 匹配数据的 ID（类比 Java: hit.getId()）
        #   hit.distance → 与查询向量的距离（越小越相似）
        #   hit.fields → 返回的标量字段字典（类比 Java: hit.getFields()）
        print(f"{i+1:<6} {hit.id:<10} {hit.distance:<12.4f} {hit.fields['category']}")
        # 注意: {hit.distance:<12.4f} — .4f 表示保留 4 位小数
        # 类比 Java: String.format("%.4f", hit.getDistance())

    print()


# ============================================================
# 【概念】标量过滤搜索
#
# 纯向量搜索 = 只看相似度，不考虑其他条件
# 带过滤的搜索 = 先按条件筛选，再在筛选结果中找最相似的向量
#
# 应用场景：
#   - "找出属于「科技」类的最相似文章"
#   - "找出最近一年发布的技术类博文"
#   - "找出属于 VIP 用户的相似商品"
# ============================================================

def search_with_filter(collection):
    """
    带标量过滤的向量搜索

    在搜索时加上 WHERE 条件，类似 SQL 中的：
      SELECT * FROM table
      WHERE category = 'tech'
      ORDER BY distance(embedding, query_vector) ASC
      LIMIT 5

    过滤表达式语法（类似 Python 表达式）：
      "category == 'tech'"
      "category in ['tech', 'science']"
      "price > 100 AND year >= 2023"
    """
    query_vector = np.random.rand(1, 128).astype(np.float32)
    k = 5

    # 过滤条件：只搜索 category == 'tech' 的向量
    filter_expr = "category == 'tech'"

    print(f"[过滤搜索] 查询条件: {filter_expr}")
    print(f"[过滤搜索] 正在搜索 Top-{k}...\n")

    # 和上面 search() 几乎一样，多了 expr 参数
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,
        expr=filter_expr,          # ← 这就是标量过滤条件（WHERE 子句）
        output_fields=["category"],
    )

    print("[过滤搜索] OK 搜索结果:")
    print(f"{'排名':<6} {'ID':<10} {'距离':<12} {'类别'}")
    print("-" * 45)
    for i, hit in enumerate(results[0]):
        print(f"{i+1:<6} {hit.id:<10} {hit.distance:<12.4f} {hit.fields['category']}")
    print()


# ============================================================
# 【Python 语法】if __name__ == "__main__"
#   这是 Python 的入口点写法
#   类比 Java: public static void main(String[] args) { ... }
#
#   __name__ 是 Python 的内置变量：
#     - 直接运行此文件时，__name__ == "__main__"
#     - 被其他文件 import 时，__name__ == 文件名
#   这样可以做到：import 不执行 main，直接运行才执行 main
# ============================================================

def main():
    """
    主函数 — 类比 Java 的 public static void main(String[] args)
    """
    print("=" * 60)
    print("  Milvus 入门 — Hello Milvus")
    print("=" * 60)
    print()

    # Step 1: 连接
    connect_milvus()

    # Step 2: 定义 Schema
    schema = create_schema()
    print("[Schema] 集合结构:")
    for field in schema.fields:
        print(f"         - {field.name}: {field.dtype.name}" +
              (f" (dim={field.params.get('dim', '')})" if field.dtype.name == "FLOAT_VECTOR" else ""))
        # --- 【Python 语法】三元表达式 ---
        # (A if condition else B) 类比 Java: condition ? A : B
        # field.params.get('dim', '') 类比 Java: params.containsKey("dim") ? params.get("dim") : ""

    print()

    # Step 3: 创建集合
    collection = create_collection(schema)

    # Step 4: 插入数据
    insert_data(collection)

    # Step 5: 创建索引
    create_index(collection)

    # Step 6: 搜索
    search(collection)

    # Step 7: 带过滤的搜索
    search_with_filter(collection)

    print("=" * 60)
    print("  OK 全部完成！")
    print("=" * 60)

    # 清理：关闭连接
    connections.disconnect("default")
    print(f"[清理] 已断开连接 'default'")


# 入口点 — 类比 Java: public static void main(String[] args)
if __name__ == "__main__":
    main()
