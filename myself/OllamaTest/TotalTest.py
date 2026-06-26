import sys
import time
import requests
from typing import List, Optional
from llama_index.llms.ollama import Ollama
from pymilvus import MilvusClient, DataType

# ==================================================================
# 🚨 你的本机专属公网穿透配置（强行写死）
# ==================================================================
MY_OLLAMA_URL = "http://427ea2ed.r23.cpolar.top"
MILVUS_PUBLIC_HOST = "6.tcp.cpolar.cn"
MILVUS_PUBLIC_PORT = 12778
MY_RERANK_URL = "http://6a1fa7f0.r23.cpolar.top"

# 🌟 强行全链路写死对齐的专属配置
COLLECTION_NAME = "green_horse"
VECTOR_FIELD = "vector_field"
TEXT_FIELD = "text_content"


def log_step(step_num, message):
    print(f"\n▲ [{time.strftime('%H:%M:%S')}] 步骤 {step_num}: {message}")
    print("-" * 60)


# ------------------------------------------------------------------
# 纯原生 Requests 获取 Ollama 向量
# ------------------------------------------------------------------
def get_bge_embedding(text: str) -> List[float]:
    url = f"{MY_OLLAMA_URL}/api/embed"
    payload = {
        "model": "bge-m3:latest",
        "input": text
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 405:
            url = f"{MY_OLLAMA_URL}/api/embeddings"
            payload = {
                "model": "bge-m3:latest",
                "prompt": text
            }
            response = requests.post(url, json=payload, timeout=30)

        response.raise_for_status()
        res_json = response.json()

        if "embeddings" in res_json:
            data = res_json["embeddings"]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return data[0]
            return data
        elif "embedding" in res_json:
            data = res_json["embedding"]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return data[0]
            return data
        else:
            raise ValueError(f"Ollama 返回了未知的 JSON 格式: {res_json}")
    except Exception as e:
        print(f"❌ 原生向量化请求失败，接口 URL: {url}。错误: {e}")
        sys.exit(1)


# ------------------------------------------------------------------
# 纯手动重排函数
# ------------------------------------------------------------------
# 自适应重排函数：支持标准 /v1/rerank 和通用 /rerank 路由
# ------------------------------------------------------------------
def direct_cpolar_rerank(query: str, documents: List[str], top_n: int = 2) -> List[str]:
    if not documents:
        return []

    # 准备测试两条最常见的 Reranker 路由路径
    urls_to_try = [
        f"{MY_RERANK_URL}/v1/rerank",  # 标准 OpenAI/TEI 风格路径
        f"{MY_RERANK_URL}/rerank"  # 常见的轻量化自建 API 路径
    ]

    payload = {
        "model": "BAAI/bge-reranker-large",
        "query": query,
        "documents": documents
    }

    response = None
    last_error = ""

    # 自动探测哪个路由可用
    for url in urls_to_try:
        print(f"[重排调度] 正在尝试向穿透端 {url} 发送 {len(documents)} 条文本片段...")
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                response = res
                print(f"🎯 成功对接重排路由: {url}")
                break
            else:
                last_error = f"HTTP {res.status_code}"
        except Exception as e:
            last_error = str(e)
            continue

    # 如果两条路径都试失败了，执行优雅降级（不崩溃，直接把初筛结果喂给大模型）
    if response is None:
        print(f"⚠️ Reranker 所有已知穿透路径均调用失败 ({last_error})，正在自动回退到 Milvus 初筛结果。")
        return documents[:top_n]

    try:
        results = response.json().get("results", [])

        # 针对部分框架返回格式不同的兼容性解析
        if not results and isinstance(response.json(), list):
            # 有些轻量框架直接返回数组格式: [{"index": 0, "relevance_score": 0.9}]
            results = response.json()

        sorted_results = sorted(results, key=lambda x: x.get("relevance_score", 0.0) or x.get("score", 0.0),
                                reverse=True)
        final_docs = [documents[res["index"]] for res in sorted_results[:top_n]]

        print(f"✅ 重排清洗成功！已筛选出最相关的 Top {len(final_docs)} 个偏正片段。")
        return final_docs
    except Exception as e:
        print(f"⚠️ Reranker 返回数据解析异常，回退到初筛结果。错误: {e}")
        return documents[:top_n]


def main():
    print("=" * 60)
    print("  🚀 联新小新 Pro 14 (MilvusClient 全自动建库对齐版 RAG) 🚀  ")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 步骤 1: 初始化大模型大脑
    # ------------------------------------------------------------------
    log_step(1, f"初始化 Ollama 模型大脑 -> 指向公网: {MY_OLLAMA_URL}")
    llm = Ollama(
        base_url=MY_OLLAMA_URL,
        model="qwen3:4b",
        request_timeout=120.0
    )
    print("✅ Ollama 文本大模型 (qwen3:4b) 挂载完毕。")

    # ------------------------------------------------------------------
    # 步骤 2: 连接并全自动构建/初始化专属测试库
    # ------------------------------------------------------------------
    log_step(2, f"全自动检查并构建知识库集合: [{COLLECTION_NAME}]")
    try:
        milvus_uri = f"http://{MILVUS_PUBLIC_HOST}:{MILVUS_PUBLIC_PORT}"
        client_milvus = MilvusClient(uri=milvus_uri)
        print(f"✅ 成功通过公网连接到 Milvus ({milvus_uri})")

        # 🤖 如果你的集合不存在，代码直接为你当场建表并灌入标准测试数据
        existing_collections = client_milvus.list_collections()
        if COLLECTION_NAME not in existing_collections:
            print(f" ℹ️ 未发现 [{COLLECTION_NAME}] 集合，正在为你初始化创建并对齐所有字段...")

            # 1. 创建 schema
            schema = client_milvus.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name=VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name=TEXT_FIELD, datatype=DataType.VARCHAR, max_length=65535)

            # 2. 配置索引
            index_params = client_milvus.prepare_index_params()
            index_params.add_index(field_name=VECTOR_FIELD, metric_type="COSINE", index_type="IVF_FLAT",
                                   params={"nlist": 128})

            # 3. 落地建表
            client_milvus.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
            print(
                f" 🎉 专属知识库集合 [{COLLECTION_NAME}] 创建成功！字段完美对齐：{VECTOR_FIELD}(1024维), {TEXT_FIELD}(文本)。")

            # 4. 塞入测试用的知识点，确保等会有东西可以检索
            print(" 📥 正在注入默认的知识库初始化语料...")
            test_texts = [
                "联想小新 Pro 14 是一款性能强劲的轻薄笔记本，搭载了高性能处理器，非常适合进行本地大模型和 AI 工程师的开发训练。",
                "大模型 RAG 技术的核心要点是通过向量数据库检索相关的背景知识库内容，然后提供给大模型进行清洗和总结，从而消除幻觉。",
                "Milvus 向量数据库能够通过 MilvusClient 提供极高并发的向量检索支持，结合重排器 Reranker 可以让答案的精度更上一层楼。"
            ]
            test_data = []
            for text in test_texts:
                vector = get_bge_embedding(text)
                test_data.append({VECTOR_FIELD: vector, TEXT_FIELD: text})

            client_milvus.insert(collection_name=COLLECTION_NAME, data=test_data)
            print(" ✅ 语料数据向量化导入完毕，知识库已就绪！")

        else:
            print(f"🎯 检查到资产库 [{COLLECTION_NAME}] 已存在，直接进入全链路调度。")

    except Exception as e:
        print(f"❌ Milvus 自动化资产配置失败: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 步骤 3: 准备查询与向量化提问
    # ------------------------------------------------------------------
    log_step(3, "执行全链路跨域调度...")
    question = "请根据知识库内容，总结一下核心要点。"
    print(f"[提问] 用户意图 -> '{question}'")

    print(" [向量化] 正在通过原生 HTTP 接口将问题转化为 bge-m3 向量...")
    query_vector = get_bge_embedding(question)
    print(f"✅ 向量化成功！真实的维度大小: {len(query_vector)}")

    # ------------------------------------------------------------------
    # 步骤 4: 用 MilvusClient 进行纯原生向量检索
    # ------------------------------------------------------------------
    print(f" [检索] 正在通过 MilvusClient 查询集合 [{COLLECTION_NAME}]...")
    try:
        search_res = client_milvus.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=3,
            output_fields=[TEXT_FIELD],
            anns_field=VECTOR_FIELD
        )

        raw_documents = []
        for hits in search_res:
            for hit in hits:
                text = hit.get('entity', {}).get(TEXT_FIELD) or hit.get('fields', {}).get(TEXT_FIELD)
                if text:
                    raw_documents.append(text)

        print(f"✅ Milvus 初筛完成，成功捞出 {len(raw_documents)} 条相似片段。")
    except Exception as e:
        print(f"❌ Milvus 检索失败！错误详情: {e}")
        sys.exit(1)

    # 2. 第二阶段：通过穿透的 Reranker 进行过滤清洗
    reranked_documents = direct_cpolar_rerank(
        query=question,
        documents=raw_documents,
        top_n=2
    )

    # 3. 拼接上下文供大模型阅读
    context_str = "\n\n".join(reranked_documents)
    prompt = f"请严格根据以下已知知识库内容，回答用户的问题。\n\n【已知内容】:\n{context_str}\n\n【用户问题】:\n{question}"

    # ------------------------------------------------------------------
    # 步骤 5: 输出最终精准答案
    # ------------------------------------------------------------------
    log_step(5, "大语言模型（Qwen3）结合清洗后上下文输出的最终答案")
    try:
        response = llm.complete(prompt)
        print(response)
    except Exception as e:
        print(f"❌ 大模型生成回复失败: {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
