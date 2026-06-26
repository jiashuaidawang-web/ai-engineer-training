from pymilvus import MilvusClient

# 创建本地数据库
client = MilvusClient("milvus_local.db")

# 创建 Collections
if (client.has_collection(collection_name="demo_collection")):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)








