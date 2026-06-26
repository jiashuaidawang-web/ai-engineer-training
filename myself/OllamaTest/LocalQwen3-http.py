from openai import OpenAI

client = OpenAI(
    base_url="http://427ea2ed.r23.cpolar.top/v1",
    api_key="ollama"
)

# 1. 开启流式传输 stream=True
resp = client.chat.completions.create(
    model="qwen3:4b",
    messages=[
        {"role": "user", "content": "用 Java 程序员能听懂的话解释 RAG"}
    ],
    stream=True  # 👈 关键点：开启流式传输，拒绝死等
)

# 2. 用循环实时打印每一个弹出的 Token
for chunk in resp:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
