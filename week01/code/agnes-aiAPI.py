import os

from openai import OpenAI
from dotenv import load_dotenv;

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

client = OpenAI(
  api_key=api_key,
  base_url=base_url
)

response = client.chat.completions.create(
  model="agnes-2.0-flash",
  messages=[
    {"role": "user", "content": "你好"}
  ]
)

print(response.choices[0].message.content)
