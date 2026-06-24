import os
from dotenv import load_dotenv

import chromadb

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    current_dir = get_current_dir()
    env_path = os.path.join(current_dir, ".env")
    load_dotenv(env_path)

    config = {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "llm_model": os.getenv("LLM_MODEL", "qwen3:4b"),
        "embed_model": os.getenv("EMBED_MODEL", "bge-m3"),
        "chroma_dir": os.path.join(current_dir, os.getenv("CHROMA_DIR", "./chroma_db")),
        "collection_name": os.getenv("CHROMA_COLLECTION", "rag_local_docs"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "80")),
        "data_dir": os.path.join(current_dir, "data"),
    }

    return config


def init_llamaindex_settings(config):
    # 1. 本地大模型：qwen3:4b
    # 负责最后根据检索内容生成答案
    Settings.llm = Ollama(
        model=config["llm_model"],
        base_url=config["ollama_base_url"],
        temperature=0.1,
        request_timeout=300000.0,
        context_window=4096,
    )

    # 2. 本地 Embedding 模型：bge-m3
    # 负责把文档切片和用户问题转成向量
    Settings.embed_model = OllamaEmbedding(
        model_name=config["embed_model"],
        base_url=config["ollama_base_url"],
    )

    # 3. 切片配置
    Settings.node_parser = SentenceSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )


def build_index(config):
    if not os.path.exists(config["data_dir"]):
        raise RuntimeError(f"data 目录不存在：{config['data_dir']}")

    documents = SimpleDirectoryReader(config["data_dir"]).load_data()
    print(f"读取到 {len(documents)} 个文档")

    # 1. 创建 Chroma 本地持久化客户端
    chroma_client = chromadb.PersistentClient(path=config["chroma_dir"])

    # 2. 创建或获取 collection
    chroma_collection = chroma_client.get_or_create_collection(
        name=config["collection_name"]
    )

    # 3. 把 Chroma 封装成 LlamaIndex 的 VectorStore
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    # 4. 创建 StorageContext
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 5. 构建索引
    # 这一行会触发：
    # Document -> Node切片 -> bge-m3生成向量 -> 写入Chroma
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"索引构建完成")
    print(f"Chroma 向量库目录：{config['chroma_dir']}")
    print(f"Collection：{config['collection_name']}")

    return index


def load_index(config):
    # 从已经持久化的 Chroma 中加载向量库
    chroma_client = chromadb.PersistentClient(path=config["chroma_dir"])

    chroma_collection = chroma_client.get_or_create_collection(
        name=config["collection_name"]
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )

    return index


def query_index(index, question):
    query_engine = index.as_query_engine(
        similarity_top_k=3,
    )

    response = query_engine.query(question)

    print("\n========== 回答 ==========")
    print(response)

    print("\n========== 来源片段 ==========")
    for i, source_node in enumerate(response.source_nodes, start=1):
        node = source_node.node
        print(f"\n--- 来源 {i} ---")
        print(f"相似度分数：{source_node.score}")
        print(f"元数据：{node.metadata}")
        print(f"文本片段：{node.text[:500]}")


def main():
    config = load_config()

    print("当前配置：")
    print(f"OLLAMA_BASE_URL = {config['ollama_base_url']}")
    print(f"LLM_MODEL       = {config['llm_model']}")
    print(f"EMBED_MODEL     = {config['embed_model']}")
    print(f"CHROMA_DIR      = {config['chroma_dir']}")
    print(f"COLLECTION      = {config['collection_name']}")


    init_llamaindex_settings(config)

    # 第一次运行：构建索引
    index = build_index(config)

    # 构建完成后，直接问一个问题测试
    question = input("\n请输入你的问题：")
    query_index(index, question)


if __name__ == "__main__":
    main()