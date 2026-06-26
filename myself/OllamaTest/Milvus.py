import time
from openai import OpenAI
from pymilvus import connections, Collection, utility

# ==========================================
# 1. 配置参数（使用你刚才 cpolar 穿透出的公网地址）
# ==========================================
OLLAMA_PUBLIC_URL = "http://427ea2ed.r23.cpolar.top/v1"  # 👈 填入你 Ollama 穿透的 https 链接
MILVUS_PUBLIC_HOST = "6.tcp.cpolar.cn"  # 👈 填入 Milvus 穿透的域名
MILVUS_PUBLIC_PORT = 12778  # 👈 填入 Milvus 穿透的 5 位随机端口
COLLECTION_NAME = "green_horse"  # 👈 你的 Milvus 集合名称（按实际修改）

print("正在初始化连接...")

# ==========================================
# 2. 通过公网连接本地 Milvus
# ==========================================
try:
    connections.connect(
        alias="default",
        host=MILVUS_PUBLIC_HOST,
        port=MILVUS_PUBLIC_PORT
    )
    print(f"✅ 成功通过公网连接到 Milvus ({MILVUS_PUBLIC_HOST}:{MILVUS_PUBLIC_PORT})")
except Exception as e:
    print(f"❌ Milvus 连接失败: {e}")
    exit()

# ==========================================
# 3. 通过公网请求 Ollama 生成测试文本的向量
# ==========================================
print("\n正在请求 Ollama 生成向量...")
try:
    client = OpenAI(base_url=OLLAMA_PUBLIC_URL, api_key="ollama")

    test_text = "这是一个关于企业知识库 RAG 的测试文本"
    resp = client.embeddings.create(
        model="bge-m3",
        input=test_text
    )
    query_vector = resp.data[0].embedding
    print(f"✅ Ollama 向量生成成功！维度: {len(query_vector)}")
except Exception as e:
    print(f"❌ Ollama 请求失败: {e}")
    connections.disconnect("default")
    exit()

# ==========================================
# 4. 将生成的向量，拿到 Milvus 中进行近邻查询 (Search)
# ==========================================
print(f"\n正在查询 Milvus 集合 [{COLLECTION_NAME}] 中的相似数据...")
try:
    # 检查集合是否存在
    if not utility.has_collection(COLLECTION_NAME):
        print(f"❌ 未在 Milvus 中找到名为 '{COLLECTION_NAME}' 的集合，请检查你的集合名称。")
    else:
        collection = Collection(COLLECTION_NAME)
        collection.load()  # 加载到内存

        # 查询配置（根据你建表时的索引类型调整，这里以 COSINE + IVF_FLAT 为例）
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        start_time = time.time()
        results = collection.search(
            data=[query_vector],  # 刚才用 Ollama 跑出来的向量
            anns_field="vector_field",  # 👈 换成你 Milvus 表里存储向量的字段名
            param=search_params,
            limit=3,  # 返回最相似的前 3 条
            output_fields=["text_content"]  # 👈 换成你想顺便查询出来的文本字段名
        )
        end_time = time.time()

        print(f"✅ 查询耗时: {(end_time - start_time) * 1000:.2f} ms")
        print("-" * 50)

        # 打印查询结果
        for hits in results:
            for hit in hits:
                print(f"👉 相似度分数 (Score): {hit.score}")
                print(f"   原始文本 (Text): {hit.entity.get('text_content')}")
                print("-" * 50)

except Exception as e:
    print(f"❌ Milvus 数据检索失败: {e}")
finally:
    # 断开连接
    connections.disconnect("default")
    print("\n测试结束，连接已释放。")