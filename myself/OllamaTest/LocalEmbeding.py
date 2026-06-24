from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

resp = client.embeddings.create(
    model="bge-m3",
    input="这是一个关于企业知识库 RAG 的测试文本"
)

print(len(resp.data[0].embedding))
print(resp.data[0].embedding[:10])