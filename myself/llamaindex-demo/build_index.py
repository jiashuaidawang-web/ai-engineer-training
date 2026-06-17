import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def main():
    # 强制读取当前文件同目录下的 .env，避免 PyCharm 工作目录不对
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, ".env")
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    llm_model = os.getenv("OPENAI_API_MODEL")

    print(f"api_key={'已读取' if api_key else None}")
    print(f"base_url={base_url}")
    print(f"llm_model={llm_model}")

    if not api_key:
        raise RuntimeError("请先在 .env 中配置 V_API_KEY")

    if not base_url:
        raise RuntimeError("请先在 .env 中配置 V_API_API_BASE")

    if not llm_model:
        raise RuntimeError("请先在 .env 中配置 V_API_MODEL")

    # 大模型：负责最终回答
    Settings.llm = OpenAI(
        model=llm_model,
        api_key=api_key,
        api_base=base_url,
        temperature=0.1,
    )

    # Embedding 模型：负责把文档转成向量
    # 注意：这里默认使用 OpenAI 官方 embedding
    # 所以 .env 里还需要有 OPENAI_API_KEY
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY") or api_key,
    )

    # 1. Loading：读取 data 目录下的文档
    data_dir = os.path.join(current_dir, "data")
    documents = SimpleDirectoryReader(data_dir).load_data()

    print(f"读取到 {len(documents)} 个文档")

    # 2. Index：构建向量索引
    index = VectorStoreIndex.from_documents(documents)

    # 3. Storage：保存到本地 storage 目录
    storage_dir = os.path.join(current_dir, "storage")
    index.storage_context.persist(persist_dir=storage_dir)

    print("索引构建完成，已保存到 storage 目录")


if __name__ == "__main__":
    main()