import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


"""" 
启动： 
uvicorn api:app --host 0.0.0.0 --port 8000
测试： 
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"青马工程包含哪些内容？"}'
  
"""

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("请先在 .env 中配置 OPENAI_API_KEY")

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

storage_context = StorageContext.from_defaults(
    persist_dir="storage"
)

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

app = FastAPI()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    response = query_engine.query(req.question)
    return AskResponse(answer=str(response))