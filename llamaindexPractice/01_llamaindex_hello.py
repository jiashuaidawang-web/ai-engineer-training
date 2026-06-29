"""
01_llamaindex_hello.py — LlamaIndex 入门：你好，世界！

这一课带你跑通 LlamaIndex 的第一个程序，理解核心概念：
- Settings：配置 LLM 和 Embedding 模型
- SimpleDirectoryReader：加载本地文档
- SummaryIndex：最简单的索引类型
- QueryEngine：查询引擎，用来回答问题

【Java 程序员速查】
  LlamaIndex 的核心思想：
    Document（文档） → Index（索引） → Query Engine（查询引擎） → 回答

  类比 Spring Boot 的链路：
    数据源（DataSource） → Repository（仓储） → Service（服务） → Controller（接口）
"""

import os
import sys

# ============================================================
# 【Python 语法】sys.path 添加
#   告诉 Python 去哪里找模块
#   类比 Java 的 classpath
# ============================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================
# 【Python 语法】dotenv 加载环境变量
#   从 .env 文件中读取 API Key，避免硬编码
#   类比 Java 的 application.properties 或环境变量
# ============================================================
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

# ============================================================
# 【Python 语法】import 说明
#   from xxx import yyy  类比 Java: import xxx.yyy;
#   大括号 import (...) 类比 Java 的多行 import 块
# ============================================================

from llama_index.core import (
    Settings,           # 全局配置 — 类比 Java 的配置类@Configuration
    SimpleDirectoryReader,  # 文档加载器 — 类比 Java 的文件读取工具类
    SummaryIndex,         # 摘要索引 — 最简单的索引类型
    StorageContext,     # 存储上下文 — 类比 Java 的上下文对象（保存状态）
    load_graph_store_from_json,  # 从 JSON 加载图存储
)
from llama_index.core.node_parser import SentenceSplitter  # 句子切分器
from llama_index.llms.openai import OpenAI  # OpenAI LLM 适配器
from llama_index.embeddings.openai import OpenAIEmbedding  # OpenAI Embedding 适配器

# ============================================================
# 第一步：配置 LLM 和 Embedding 模型
# ============================================================

# --- 【Python 语法】Settings 对象 ---
# Settings 是 LlamaIndex 的全局配置对象
# 类比 Java:
#   Settings.setLLM(new OpenAILLM());
#   Settings.setEmbedding(new OpenAIEmbedding());
# LlamaIndex 用全局单例模式管理配置，所有地方都能访问

# 配置 LLM（大语言模型）
# model="gpt-3.5-turbo" — 指定使用 GPT-3.5 Turbo 模型
# temperature=0.1 — 控制创造性，0.1 表示尽量保守、准确
# temperature 类比 Java 中的随机种子，值越低输出越稳定
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取 API Key
    base_url=os.getenv("OPENAI_API_BASE"),  # API 地址
)

# 配置 Embedding 模型
# embed_model="text-embedding-ada-002" — OpenAI 的 embedding 模型
Settings.embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

print("=" * 60)
print("  LlamaIndex 入门 — Hello, LlamaIndex!")
print("=" * 60)
print(f"\n[配置] LLM: {Settings.llm.model}")
print(f"[配置] Embedding: text-embedding-ada-002")
print(f"[配置] Temperature: {Settings.llm.temperature}")


# ============================================================
# 第二步：加载文档
# ============================================================

def load_documents():
    """
    加载本地文档

    SimpleDirectoryReader 会自动扫描指定目录下的所有文件，
    并根据文件扩展名选择合适的解析器。

    支持的格式：
      .txt, .pdf, .md, .docx, .pptx, .xlsx, .epub, .html, .xml
      .jpg, .png, .gif（图像 OCR）

    类比 Java:
      List<Document> docs = new SimpleDirectoryReader("./data")
          .loadDocuments();

    返回：
      List[Document] — 文档列表
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "llamaindex-demo", "data")

    # --- 【Python 语法】os.path.join ---
    # 拼接路径，自动处理不同系统的分隔符（Windows 用 \，Linux 用 /）
    # 类比 Java: Paths.get(base, relative).toString()

    if not os.path.exists(data_dir):
        print(f"[加载] 数据目录不存在: {data_dir}")
        print(f"[加载] 请先将文档放入 llamaindex-demo/data/ 目录")
        return []

    print(f"\n[加载] 正在从 '{data_dir}' 加载文档...")

    # --- 【Python 语法】SimpleDirectoryReader ---
    # 构造器链式调用，类比 Java 的 Builder 模式
    reader = SimpleDirectoryReader(data_dir)

    # --- 【Python 语法】.load_documents() ---
    # 加载所有文档，返回 Document 对象列表
    # 类比 Java: reader.loadDocuments()
    documents = reader.load_documents()

    print(f"[加载] ✓ 共加载 {len(documents)} 个文档")
    for i, doc in enumerate(documents):
        # --- 【Python 语法】doc 对象的属性 ---
        # doc.id_ — 文档 ID
        # doc.text — 文档内容
        # doc.metadata — 文档元数据（文件名、路径等）
        print(f"    文档 {i+1}: {doc.metadata.get('file_name', 'unknown')} "
              f"(长度: {len(doc.text)} 字符)")

    return documents


# ============================================================
# 第三步：文本分块（Chunking）
# ============================================================

def chunk_documents(documents):
    """
    将文档切分为小块

    为什么需要分块？
    - LLM 有上下文窗口限制（GPT-3.5 约 4096 token）
    - Embedding 模型也有输入长度限制
    - 小块的检索质量更高

    SentenceSplitter 的策略：
    1. 先按句子分割（遇到句号、问号、感叹号就分）
    2. 如果句子太长，再按单词分割
    3. 合并短句子直到达到 chunk_size

    类比 Java:
      TokenTextSplitter splitter = new TokenTextSplitter(500, 50);
      List<TextSplit> chunks = splitter.splitDocuments(documents);

    参数：
      chunk_size: 每个块的最大 token 数（默认 1024）
      chunk_overlap: 相邻块之间的重叠 token 数（默认 20）
      separator: 分割符（默认 " " 空格）

    返回：
      List[Document] — 切分后的块（每个块也是一个 Document）
    """
    print("\n[分块] 正在对文档进行分块...")

    # --- 【Python 语法】SentenceSplitter ---
    # 按句子边界分割文本
    splitter = SentenceSplitter(
        chunk_size=512,    # 每个块最多 512 个 token
        chunk_overlap=50,  # 相邻块重叠 50 个 token（防止跨句子的语义断裂）
    )

    # --- 【Python 语法】.split_documents() ---
    # 对文档列表进行分块，返回分块后的文档列表
    # 类比 Java: splitter.splitDocuments(documents)
    nodes = splitter.split_documents(documents)

    print(f"[分块] ✓ 共切分为 {len(nodes)} 个块")
    if nodes:
        print(f"    第一个块: \"{nodes[0].text[:80]}...\"")
        print(f"    最后一个块: \"{nodes[-1].text[:80]}...\"")

    return nodes


# ============================================================
# 第四步：构建索引
# ============================================================

def build_index(nodes):
    """
    构建 SummaryIndex

    索引类型选择：
    - SummaryIndex: 将所有文本拼在一起，适合小文档
    - VectorStoreIndex: 向量化存储，支持相似度搜索（最常用）
    - KeywordTableIndex: 关键词表索引
    - TreeIndex: 树形索引

    这里先用最简单的 SummaryIndex 做入门，后续课程会学 VectorStoreIndex

    类比 Java:
      SummaryIndex index = new SummaryIndex(nodes);
      index.build();

    返回：
      SummaryIndex — 构建好的索引对象
    """
    print("\n[索引] 正在构建 SummaryIndex...")

    # --- 【Python 语法】SummaryIndex.from_documents() ---
    # 从节点列表构建索引（一步完成：分块 + 索引化）
    # 类比 Java: SummaryIndex index = SummaryIndex.fromDocuments(nodes);
    index = SummaryIndex.from_documents(nodes)

    print(f"[索引] ✓ 索引构建完成，共 {index.index_struct.num_nodes()} 个节点")
    return index


# ============================================================
# 第五步：创建查询引擎
# ============================================================

def create_query_engine(index):
    """
    创建查询引擎

    查询引擎是 LlamaIndex 的核心接口，负责：
    1. 接收用户问题
    2. 从索引中检索相关信息
    3. 将信息 + 问题 发送给 LLM
    4. LLM 生成回答

    类比 Java:
      QueryEngine engine = index.asQueryEngine();
      // 等同于:
      // RetrievalQueryEngine engine = RetrievalQueryEngine.fromIndex(index);

    返回：
      QueryEngine — 查询引擎对象
    """
    print("\n[查询引擎] 正在创建查询引擎...")

    # --- 【Python 语法】.as_query_engine() ---
    # 将索引转换为查询引擎
    # 类比 Java: index.asQueryEngine()
    query_engine = index.as_query_engine()

    print("[查询引擎] ✓ 查询引擎已创建")
    return query_engine


# ============================================================
# 第六步：提问并获取回答
# ============================================================

def ask_questions(query_engine):
    """
    使用查询引擎回答问题

    这是整个流程的最后一步：用户提问 → 系统回答

    类比 Java:
      QueryResponse response = queryEngine.query("你的问题");
      System.out.println(response.getResponse());

    参数：
      query_engine — 查询引擎对象

    返回：
      QueryResponse — 查询结果
    """
    print("\n" + "=" * 60)
    print("  开始提问...")
    print("=" * 60)

    # --- 【Python 语法】input() ---
    # 从控制台读取用户输入
    # 类比 Java: Scanner sc = new Scanner(System.in); String line = sc.nextLine();
    questions = [
        "你能总结一下这些文档的内容吗？",
        "文档中提到了哪些关键技术？",
        "请列出文档中出现的所有术语。",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 问题: {question}")
        print("-" * 40)

        # --- 【Python 语法】query_engine.query() ---
        # 发送查询请求，返回 QueryResponse 对象
        # 类比 Java: QueryResponse response = engine.query(question);
        response = query_engine.query(question)

        # --- 【Python 语法】response.response ---
        # QueryResponse 对象的 response 属性包含 LLM 的回答
        # 类比 Java: response.getResponse()
        print(f"回答: {response.response}")

        # --- 【Python 语法】response.source_nodes ---
        # 检索到的相关节点（用于调试和追溯）
        # 类比 Java: response.getSourceNodes()
        if hasattr(response, 'source_nodes'):
            print(f"参考了 {len(response.source_nodes)} 个文档块")

    return response


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数 — 类比 Java 的 public static void main(String[] args)

    执行流程：
    1. 配置 LLM 和 Embedding
    2. 加载文档
    3. 文本分块
    4. 构建索引
    5. 创建查询引擎
    6. 提问并获取回答
    """
    print("\n[1/6] 配置模型...")
    # 配置已在文件顶部完成

    print("\n[2/6] 加载文档...")
    documents = load_documents()
    if not documents:
        print("[错误] 没有加载到文档，退出")
        return

    print("\n[3/6] 文本分块...")
    nodes = chunk_documents(documents)

    print("\n[4/6] 构建索引...")
    index = build_index(nodes)

    print("\n[5/6] 创建查询引擎...")
    query_engine = create_query_engine(index)

    print("\n[6/6] 提问...")
    ask_questions(query_engine)

    print("\n" + "=" * 60)
    print("  OK 全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    # --- 【Python 语法】if __name__ == "__main__" ---
    # 这是 Python 的标准入口写法
    # 直接运行此文件时，__name__ == "__main__"
    # 被其他文件 import 时，__name__ == 文件名
    # 这样可以做到：import 不执行 main，直接运行才执行 main
    # 类比 Java: public static void main(String[] args) { ... }
    main()
