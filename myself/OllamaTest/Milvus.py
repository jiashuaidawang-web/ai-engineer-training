import time
from openai import OpenAI
from pymilvus import MilvusClient  # 👈 引入最新的客户端类

# ==========================================
# 1. 配置参数
# ==========================================
OLLAMA_PUBLIC_URL = "http://427ea2ed.r23.cpolar.top"  # 你的 Ollama 穿透链接
MILVUS_PUBLIC_HOST = "6.tcp.cpolar.cn"
MILVUS_PUBLIC_PORT = 12778
COLLECTION_NAME = "green_horse"  # 确保你在 Milvus 里的表名和这个一致

print("正在初始化连接...")

# ==========================================
# 2. 用新版 MilvusClient 连接（自动管理连接和释放）
# ==========================================
try:
    # 新版直接拼成一个 uri 字符串进行连接
    milvus_uri = f"http://{MILVUS_PUBLIC_HOST}:{MILVUS_PUBLIC_PORT}"
    client_milvus = MilvusClient(uri=milvus_uri)
    print(f"✅ 成功通过公网连接到 Milvus ({milvus_uri})")
except Exception as e:
    print(f"❌ Milvus 连接失败: {e}")
    exit()

# ==========================================
# 3. 通过公网请求 Ollama 生成向量
# ==========================================
print("\n正在请求 Ollama 生成向量...")
try:
    client_openai = OpenAI(base_url=OLLAMA_PUBLIC_URL, api_key="ollama")
    test_text = "这是一个关于企业知识库 RAG 的测试文本"
    resp = client_openai.embeddings.create(model="bge-m3", input=test_text)
    query_vector = resp.data[0].embedding
    print(f"✅ Ollama 向量生成成功！维度: {len(query_vector)}")
except Exception as e:
    print(f"❌ Ollama 请求失败: {e}")
    exit()

# ==========================================
# 4. 新版 Milvus 数据查询
# ==========================================
print(f"\n正在查询 Milvus 集合 [{COLLECTION_NAME}] 中的相似数据...")
try:
    # 使用新版 client_milvus.has_collection 判断
    if not client_milvus.has_collection(collection_name=COLLECTION_NAME):
        print(f"❌ 未在 Milvus 中找到名为 '{COLLECTION_NAME}' 的集合，请检查你的集合名称。")
    else:
        start_time = time.time()
        # 新版 search 接口极其精简，不需要手动 load 集合
        results = client_milvus.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=3,
            anns_field="vector_field",  # 你的向量字段名
            output_fields=["text_content"]  # 你的文本字段名
        )
        end_time = time.time()

        print(f"✅ 查询耗时: {(end_time - start_time) * 1000:.2f} ms")
        print("-" * 50)

        # 打印查询结果
        for hits in results:
            for hit in hits:
                print(f"👉 相似度分数 (Score): {hit['distance']}")
                print(f"   原始文本 (Text): {hit['entity'].get('text_content')}")
                print("-" * 50)

except Exception as e:
    print(f"❌ Milvus 数据检索失败: {e}")
finally:
    print("\n测试结束。")