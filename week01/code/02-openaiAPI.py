import os
from dotenv import load_dotenv
from openai import OpenAI
import inspect

print(inspect.signature(OpenAI))

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')
print(f"-- debug -- openai api key is {api_key[0:10]} api_base {base_url}")


# client = OpenAI(
#     base_url=base_url,
#     api_key=api_key
# )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)


# 正常会输出结果：Hello! It's great to see you. How can I assist you today?