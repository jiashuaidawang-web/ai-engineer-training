from openai import OpenAI

client = OpenAI(
    base_url="http://63d1da43.r18.vip.cpolar.cn/v1",
    api_key="ollama"
)

resp = client.embeddings.create(
    model="bge-m3",
    input="这是一个关于企业知识库 RAG 的测试文本"
)

print(len(resp.data[0].embedding))
print(resp.data[0].embedding[:10])