import os
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("请先在 .env 中配置 OPENAI_API_KEY")

    Settings.llm = OpenAI(
        model=os.getenv("OPENAI_API_MODEL"),
        temperature=0.1
    )

    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )

    # 从 storage 加载已经构建好的索引
    storage_context = StorageContext.from_defaults(
        persist_dir="storage"
    )

    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    while True:
        question = input("\n请输入问题，输入 exit 退出：")

        if question.lower() in ["exit", "quit"]:
            break

        response = query_engine.query(question)

        print("\n回答：")
        print(response)


if __name__ == "__main__":
    main()