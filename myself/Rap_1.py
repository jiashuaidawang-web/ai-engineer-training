# 配置通义千问大模型和文本向量模型
import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

# 先load配置文件
load_dotenv()

api_key_1 = os.getenv("DASHSCOPE_API_KEY")
print("api_key------"+api_key_1)

Settings.llm = OpenAILike(
  model="qwen-plus",
  api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
  api_key=api_key_1,
  is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
  model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
  embed_batch_size=6,
  embed_input_length=8192
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("怎么休事假？")
print(response)
