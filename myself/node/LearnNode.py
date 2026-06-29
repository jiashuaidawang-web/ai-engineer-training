"""
===============================================================================
 学习笔记：RAG 系统中的 Node（节点）与句子/文本切片（Chunking）
===============================================================================

【为什么学这个？—— 从 Java 视角理解】
----------------------------------------
在 RAG（检索增强生成）系统中，你需要把一篇长文档（比如几百页的制度文件）
切成小块，存入向量数据库。每一块就叫一个 "Node"（节点）。

类比 Java 项目：
  Document   ≈ 你读入的一个完整文件（比如一份 PDF 的全部内容读成一个 String）
  Node       ≈ 把 String 切成若干段，每一段就是一个 Node（类似 String.split() 的结果，但更智能）
  Splitter   ≈ 切分的"算法"（类似 Java 里的各种 Tokenizer）
  Chunk      ≈ 切出来的每一小段文本

【这个文件的学习路线】
  第 1 节：Python 基础速成（给 Java 程序员看的）
  第 2 节：什么是 Document / Node / TextNode
  第 3 节：5 种切片方法对比（手动切分 → Token 切分 → 句子切分 → 语义切分 → Markdown 切分）
  第 4 节：完整实战 — 把一个真实文档切成 Node 并打印

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate        # Linux/Mac 激活虚拟环境
    python ../myself/node/LearnNode.py
"""
from projects.project5_2.src import protos
from week02.p16_tokenizers import model_name


# ============================================================================
# 第 1 节：Python 基础速成（面向 Java 程序员）
# ============================================================================
#
# Python 和 Java 的关键区别：
#
# 1. 缩进代替大括号
#    Java:  public void foo() { System.out.println("hi"); }
#    Python: def foo():              # 注意末尾的冒号
#              print("hi")            # 前面 4 个空格表示"在大括号里面"
#
# 2. 动态类型 vs 静态类型
#    Java:  String name = "hello";    // 必须先声明类型
#    Python: name = "hello"           # Python 自己猜类型（str）
#            name: str = "hello"      # 可以写类型提示，但不是强制的
#
# 3. 没有 public/private/protected
#    Java:  private int age;
#    Python: _age          # 单下划线 = "别动我"的约定
#            __age         # 双下划线 = Python 会"名字改写"（类似 private）
#
# 4. 没有分号
#    一行一条语句，不需要分号结尾
#
# 5. 数组/List
#    Java:  List<String> list = Arrays.asList("a", "b");
#    Python: lst = ["a", "b"]           # 列表（可变数组）
#            tup = ("a", "b")           # 元组（不可变数组）
#
# 6. Map/字典
#    Java:  Map<String, Integer> map = new HashMap<>();
#           map.put("a", 1);
#    Python: d = {"a": 1}              # 字典（就是 Java 的 Map）
#            d["a"]                    # 取值（不用 get()）
#            d.get("a", 0)             # 带默认值（类似 getOrDefault）
#
# 7. 字符串格式化
#    Java:  String s = String.format("Hello %s", name);
#    Python: s = f"Hello {name}"        # f-string（最常用）
#
# 8. 循环
#    Java:  for (String item : list) { ... }
#    Python: for item in lst: ...       # 注意冒号
#
# 9. 条件判断
#    Java:  if (x > 0) { ... } else { ... }
#    Python: if x > 0: ...
#            else: ...                  # 没有 else if，用 elif
#
# 10. 空值
#     Java:  null
#     Python: None                     # 判空用 "if x is None:"
#
# 11. 类（Class）
#     Java:
#       public class Person {
#         private String name;
#         public Person(String name) { this.name = name; }
#       }
#     Python:
#       class Person:
#           def __init__(self, name):  # __init__ = 构造函数
#               self.name = name       # self = this
#
# 12. 列表推导式（Java Stream 的简洁版）
#     Java:  list.stream().filter(s -> s.length() > 3).toList()
#     Python: [s for s in lst if len(s) > 3]
#
# 13. 导入（Import）
#     Java:  import java.util.List;
#     Python: from typing import List   # 类型提示用
#             import json               # 直接用
#
# 14. 主函数入口
#     Java:  public static void main(String[] args)
#     Python: if __name__ == "__main__":
#                 main()
#
# 15. 异常处理
#     Java:  try { ... } catch (Exception e) { ... } finally { ... }
#     Python: try: ...
#             except Exception as e: ...
#             finally: ...
#
# 16. 资源管理（try-with-resources）
#     Java:  try (Connection conn = ds.getConnection()) { ... }
#     Python: with conn: ...           # 上下文管理器，退出时自动 close
#
# 17. 枚举（Enum）
#     Java:  enum Color { RED, GREEN }
#     Python: 用 class 继承 Enum，或者简单场景直接用字符串常量
#
# 18. @dataclass（Lombok @Data 的等价物）
#     Java:  @Data class Point { int x; int y; }
#     Python:
#       @dataclass
#       class Point:
#           x: int
#           y: int
#     自动生成 __init__, __repr__, __eq__, __hash__


# ============================================================================
# 第 2 节：Python 标准库速查（哪些库像 Java 的什么包）
# ============================================================================
#
# Python 标准库          Java 等价包              用途
# -------------------    --------------------     -------------------
# os                     java.nio.file            文件系统操作
# sys                    java.lang.System         系统级（stdin/stdout/path）
# json                   javax.json / Jackson     JSON 序列化/反序列化
# logging                java.util.logging        日志框架
# re                     java.util.regex          正则表达式
# datetime               java.time                日期时间
# collections            java.util                集合工具（Counter/Deque/OrderedDict）
# functools              java.util.function       函数式工具（lru_cache 类似 memoize）
# pathlib                java.nio.file.Path       路径操作（比 os.path 更优雅）
# typing                 (Java 天生有类型)         类型提示（非强制）
# abc                    java.lang.abstract       抽象基类
# dataclasses            Lombok @Data             数据类
# itertools              IntStream/Stream         迭代器工具
# contextlib             (try-with-resources)      with 语句的支持
# unittest               junit                    单元测试
# subprocess             ProcessBuilder           执行外部进程
# threading              java.lang.Thread         多线程
# concurrent.futures     java.util.concurrent     线程池/Future
# http.client            java.net.HttpURLConnection HTTP 请求
# urllib.request         java.net.URL             URL 操作
# hashlib                java.security.MessageDigest  哈希（MD5/SHA）
# base64                 javax.xml.bind.DatatypeConverter  Base64 编解码
# copy                   java.util.Collections    深拷贝/浅拷贝
# math                   java.lang.Math           数学运算
# functools.reduce       Stream.reduce()          归约操作
#
# Python 第三方库        Java 等价库              用途
# -------------------    --------------------     -------------------
# requests               OkHttp/HttpClient        HTTP 客户端
# numpy                  (无直接等价)             高性能数值计算
# pandas                 (无直接等价)             数据分析表格
# llama-index            (RAG 专属框架)           本项目的核心库
# neo4j                  neo4j-java-driver        Neo4j 图数据库驱动
# openai                 (SDK 独有)               OpenAI API 调用
# dashscope              (SDK 独有)               阿里云通义千问 API
# pytest                 JUnit                    测试框架
# fastapi                Spring Boot              Web 框架
# uvicorn                Netty                    ASGI 服务器


# ============================================================================
# 第 3 节：核心概念 — Document / Node / TextNode
# ============================================================================
#
# 在 LlamaIndex 框架中（这是做 RAG 的主流框架之一）：
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                     RAG 数据处理管线                                │
# │                                                                     │
# │  原始文档 (.pdf/.txt/.docx)                                          │
# │       │                                                             │
# │       ▼                                                             │
# │  Document（完整文档，可能几十万字符）                                  │
# │       │                                                             │
# │       ▼  ←── 这就是"切片/Chunking"，核心步骤！                       │
# │  Node Parser / Splitter（切分器）                                     │
# │       │                                                             │
# │       ▼                                                             │
# │  Node / TextNode（切好的小块，每块几千字符）                           │
# │       │                                                             │
# │       ▼                                                             │
# │  向量化（Embedding）→ 存入向量数据库                                   │
# │       │                                                             │
# │       ▼                                                             │
# │  用户提问 → 向量相似度搜索 → 找到最相关的 Node → 喂给 LLM              │
# └─────────────────────────────────────────────────────────────────────┐
#
# 【类比 Java】
#   Document  =  你读入的一整份文件内容（String）
#   Splitter  =  切分算法（类似 String.split()，但更聪明）
#   Node      =  切分后的每一段（String[] 里的一个元素，但带额外元数据）
#
# 【为什么需要 Node？】
#   1. LLM 的上下文窗口有限（比如 qwen-plus 最多 32K token）
#   2. 向量相似度搜索需要固定长度的文本段
#   3. 检索时需要知道"这段来自原文的哪个位置"（元数据）
#
# 【Document 的结构（类比 Java Bean）】
#   Document {
#       id_:           String          // 唯一 ID（UUID）
#       text:          String          // 文档全文
#       metadata:      Dict            // 元数据（文件名、作者、创建时间等）
#       embedding:     List[float]     // 向量表示（可选）
#   }
#
# 【TextNode 的结构（类比 Java Bean）】
#   TextNode {
#       id_:           String          // 节点唯一 ID
#       text:          String          // 这一小段文本
#       metadata:      Dict            // 继承了 Document 的元数据
#       embedding:     List[float]     // 向量表示
#       ref_doc_id:      String          // 所属的 Document ID
#   }
#
# 【Node 和 Document 的关系】
#   1 个 Document  →  N 个 Node（N 取决于切片大小）
#   就像 1 本书  →  几百页（每页就是一个 Node）


# ============================================================================
# 第 4 节：5 种切片方法详解
# ============================================================================

# ---------------------------------------------------------------------------
# 方法 1：手动切片（最基础，理解原理）
# ---------------------------------------------------------------------------
#
# 类比 Java：
#   String sub = fullText.substring(0, 8);
#
# Python 字符串切片语法：
#   text[start:end:step]
#   - start:  起始索引（包含），默认 0
#   - end:    结束索引（不包含），默认到末尾
#   - step:   步长，默认 1
#
# 示例：
#   text = "Hello World"
#   text[0:5]    → "Hello"    (Java: text.substring(0, 5))
#   text[6:]     → "World"    (Java: text.substring(6))
#   text[:5]     → "Hello"    (Java: text.substring(0, 5))
#   text[-5:]    → "World"    (最后 5 个字符)
#   text[::2]    → "HloWrd"   (每隔一个取一个)
#   text[::-1]   → "dlroW olleH" (反转)

def demo_manual_split():
    """
    手动切片演示

    这是最原始的方式，直接指定起止索引。
    实际项目中不会这么用，但理解它有助于明白后面自动切片的原理。
    """
    text = "CEO 可以直接请假，无需向直接领导汇报"

    # Python 字符串切片：text[start:end]
    # start 包含，end 不包含（和 Java substring 一样）
    part1 = text[0:8]  # "CEO 可以直接"
    part2 = text[9:16]  # "假，无需向直接"

    print(f"手动切片 - 第1段: '{part1}'")
    print(f"手动切片 - 第2段: '{part2}'")
    print()

    text = "123456789"
    # 要到4
    text1 = text[0:4]
    # 要6以后
    text2 = text[-4:]

    print(f"text1: {text1}")
    print(f"text2: {text2}")


# ---------------------------------------------------------------------------
# 方法 2：Token 切片（最常用，按 token 数量切）
# ---------------------------------------------------------------------------
#
# 什么是 Token？
#   LLM 不处理"字符"，处理的是"token"。
#   英文：1 token ≈ 0.75 单词
#   中文：1 token ≈ 1~2 字
#   所以按 token 切比按字数切更准确。
#
# TokenTextSplitter 的参数：
#   chunk_size=512     → 每个节点最多 512 个 token
#   chunk_overlap=20   → 相邻节点重叠 20 个 token（防止边界信息丢失）
#   separator="\n"     → 优先在换行处断开（保持段落完整）
#
# 类比 Java：
#   // 伪代码
#   List<String> chunks = tokenizer.split(document.text, maxTokens=512, overlap=20);

def demo_token_split():
    """
    Token 切片演示

    使用 LlamaIndex 内置的 TokenTextSplitter。
    这是 RAG 中最常用的切片方式，因为它按 token 数切，
    保证每段文本的 token 数不会超出 LLM 的限制。
    """
    # 导入需要的类
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core import Document

    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core import Document

    # 创建一个模拟文档
    doc = Document(
        text="""
        第七条 事假
        1. 员工因私事必须本人处理的，可申请事假。
        2. 事假需提前申请并获直属主管批准，紧急情况可事后补办手续。
        3. 事假为无薪假，按日扣除相应工资。
        4. 每月事假原则上不超过3天，全年累计不超过15天，特殊情况需经人力资源部及公司领导审批。
        """,
        metadata={"title": "员工休假制度", "chapter": "第七条"}
    )

    doc = Document(
        text="""
        红楼梦 贾奶奶进贾府
        """,
        metadata={
            "title": "贾府",
            "chapter": "第七条"
        }
    )

    # 创建切分器：每块 64 token，重叠 4 token，优先在换行处切
    splitter = TokenTextSplitter(
        chunk_size=64,  # 每个节点的最大 token 数
        chunk_overlap=4,  # 相邻节点的 token 重叠数（防止边界信息丢失）
        separator="\n"  # 优先在这个字符处断开
    )

    # 执行切分：把一个 Document 切成多个 Node
    # 类比 Java: List<Node> nodes = splitter.split(doc);
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"原始文档长度: {len(doc.text)} 字符")
    print(f"切分成 {len(nodes)} 个节点:\n")

    for i, node in enumerate(nodes):
        print(f"--- 节点 {i + 1} ---")
        print(f"  文本: {node.text}")
        print(f"  元数据: {dict(node.metadata)}")
        print(f"  所属文档ID: {node.ref_doc_id}")
        print()

## 使用tokenSplite

tokenSplitter = TokenTestSplitter(
    chunk_size=64,
    chunk_overlap=6,
    separator="\n",
)
nodes = tokenSplitter.get_nodes_from_documents([doc])
print(f"原始文档长度: {len(doc.text)} 字符")
print(f"切分后 {len(nodes)} 个节点  \n")

for i,node in enumerate(nodes):
    print(f"--节点 {i+1} 个")
    print(node.text)



# ---------------------------------------------------------------------------
# 方法 3：句子切片（按语义完整性切）
# ---------------------------------------------------------------------------
#
# SentenceSplitter 的特点：
#   - 先按句号/问号/感叹号切分成"句子"
#   - 再把相邻句子拼在一起，直到达到 chunk_size
#   - 保证每个节点至少包含完整的句子
#
# 适用场景：对话体文本、FAQ、问答对
#
# chunk_size=512  → 每个节点最多 512 个 token
# chunk_overlap=50 → 相邻节点重叠 50 个 token

def demo_sentence_split():
    """
    句子切片演示

    SentenceSplitter 会先识别句子边界（。！？等），
    然后把完整的句子组合成块。
    优点是每句话都是完整的，不会被拦腰切断。
    """
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Document

    from llama_index.core.node_parse import SentenceSplitterNodeParser
    from llama_index.core import Document


    doc = Document(
        text="""
        公司实行每日八小时工作制。
        上午工作时间为九时至十二时，中午休息一小时。
        下午工作时间为十三时三十分至十七时三十分。
        员工每周享有两天休息日，通常为周六和周日。
        因工作需要安排加班的，应优先安排补休。
        无法安排补休的，按国家规定支付加班工资。
        """,
        metadata={"department": "人力资源部"}
    )

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    splitter = SentenceSplitter(
        chun_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_node_from_documents([doc])
    for i,node in enumerate(nodes):
        print(f"这是第 {i+1} 个节点")
        print(f"node的内容是 {node.text}")

    nodes = splitter.get_nodes_from_documents([doc])

    print(f"句子切片 — 切分成 {len(nodes)} 个节点:\n")

    for i, node in enumerate(nodes):
        print(f"--- 节点 {i + 1} ---")
        print(f"  文本: {node.text.strip()}")
        print(f"  元数据: {dict(node.metadata)}")
        print()


# ---------------------------------------------------------------------------
# 方法 4：语义切片（最智能，按语义相似度切）
# ---------------------------------------------------------------------------
#
# SemanticSplitterNodeParser 的原理：
#   1. 先把文本逐句切分
#   2. 对每句话做向量嵌入（embedding）
#   3. 比较相邻句子的向量相似度
#   4. 相似度低的交界处就是"语义断点"，在这里切分
#
# 类比：就像一个懂中文的人在读文章，知道哪里话题变了就该分段。
#
# 优点：切出来的块语义连贯，检索质量最高
# 缺点：需要调用 embedding 模型，速度较慢，有成本

def demo_semantic_split():
    """
    语义切片演示

    这是最智能的切片方式。它会：
    1. 把文本拆成句子
    2. 对每个句子做向量嵌入（embedding）
    3. 计算相邻句子的向量相似度
    4. 相似度突然下降的地方 = 话题切换 = 切分点

    优点：切出来的块语义最连贯
    缺点：需要调用 Embedding 模型（有 API 成本 + 速度慢）
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.core import Settings
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

    from llama_index.core.node_parse import SemanticSplitterNodeParser
    from llama_index.core import Document
    from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModes


    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192,
    )



    # 配置 embedding 模型（阿里云通义千问）
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )

    doc = Document(
        text="""
        第一章 总则
        为规范公司员工休假管理，保障员工合法权益，制定本制度。
        本制度适用于公司所有正式员工。

        第二章 休假类型
        员工可享受以下假期：法定节假日、带薪年休假、病假、事假、婚假、产假等。
        法定节假日共11天，包括元旦1天、春节3天、国庆节3天等。

        第三章 年假规定
        工作满1年不满10年的，年休假5天。
        满10年不满20年的，年休假10天。
        满20年及以上的，年休假15天。

        第四章 病假规定
        员工因病可申请病假，需提供二级以上医院诊断证明。
        病假30天以内按基本工资80%发放。
        """,
        metadata={"title": "员工休假管理制度"}
    )


    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95
    )

    nodes = splitter.get_nodes_from_documents([doc])

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,  # 前后各看几句来做相似度比较
        breakpoint_percentile_threshold=95  # 相似度差值超过 95% 分位才切分（越严格切得越少）
    )

    nodes = splitter.get_nodes_from_documents([doc])

    print(f"语义切片 — 切分成 {len(nodes)} 个节点:\n")

    for i, node in enumerate(nodes):
        print(f"--- 节点 {i + 1} ---")
        print(f"  文本: {node.text.strip()[:80]}...")  # 只显示前80字符
        print(f"  元数据: {dict(node.metadata)}")
        print()


# ---------------------------------------------------------------------------
# 方法 5：Markdown 切片（针对 Markdown 格式文档）
# ---------------------------------------------------------------------------
#
# MarkdownNodeParser 的特点：
#   - 按 Markdown 的标题层级（# ## ###）来切分
#   - 每个章节成为一个独立的 Node
#   - 保留 Markdown 格式作为元数据
#
# 适用场景：技术文档、Wiki、README、API 文档

def demo_markdown_split():
    """
    Markdown 切片演示

    按 Markdown 的标题层级（# ## ###）切分。
    每个章节成为一个 Node，天然保持语义完整。
    """
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.core import Document

    doc = Document(
        text="""
        # API 接口文档

        ## 1. 用户接口

        ### 1.1 获取用户信息
        GET /api/users/{id}
        返回指定用户的详细信息。

        ### 1.2 更新用户信息
        PUT /api/users/{id}
        更新用户的基本信息。

        ## 2. 订单接口

        ### 2.1 创建订单
        POST /api/orders
        创建一个新的订单。

        ### 2.2 查询订单
        GET /api/orders/{id}
        查询订单的详细信息。
        """,
        metadata={"type": "documentation"}
    )

    splitter = MarkdownNodeParser()
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"Markdown 切片 — 切分成 {len(nodes)} 个节点:\n")

    for i, node in enumerate(nodes):
        print(f"--- 节点 {i + 1} ---")
        print(f"  文本: {node.text.strip()[:60]}...")
        print(f"  元数据: {dict(node.metadata)}")
        print()


# ============================================================================
# 第 5 节：Node 的属性详解（类比 Java Bean 的 getter）
# ============================================================================

def demo_node_attributes():
    """
    展示 TextNode 的所有重要属性

    每个 Node 就像 Java 里的一个 POJO，有很多 getter 方法：
      node.text       → 文本内容（String）
      node.metadata   → 元数据（Dict，类似 Java 的 Map<String, Object>）
      node.ref_doc_id   → 所属文档 ID
      node.id_        → 节点自身 ID
      node.get_text() → 同 text，但会拼接 metadata 前缀
      node.set_text() → 设置文本
      node.split()    → 把自己再切分成更小的 Node
    """
    from llama_index.core.schema import TextNode

    # 创建 TextNode（类比 Java: new TextNode(text, metadata)）
    node = TextNode(
        text="这是一段示例文本，用于演示 Node 的属性。",
        metadata={
            "source": "test_file.txt",
            "page": 1,
            "section": "演示"
        },
        ref_doc_id="doc-001"
    )

    print("=== TextNode 属性一览 ===\n")
    print(f"node.id_       = {node.id_}")
    print(f"node.text      = {node.text}")
    print(f"node.ref_doc_id  = {node.ref_doc_id}")
    print(f"node.metadata  = {dict(node.metadata)}")
    print(f"node.get_text() = {node.get_text()}")  # 会拼接 metadata 前缀
    print()


# ============================================================================
# 第 6 节：Python 常用内建函数速查（Java 程序员必看）
# ============================================================================

def demo_python_builtins():
    """
    Python 日常开发中最常用的内建函数

    这些函数不需要 import，直接使用：
    """

    # ---- len() — 获取长度 ----
    # 类比 Java: list.size() / string.length() / array.length
    lst = [1, 2, 3, 4, 5]
    txt = "Hello"
    print(f"len([1,2,3,4,5]) = {len(lst)}")  # 5
    print(f'len("Hello") = {len(txt)}')  # 5

    # ---- range() — 生成数字序列 ----
    # 类比 Java: for (int i = 0; i < 10; i++)
    for i in range(5):  # 0, 1, 2, 3, 4（不含 5）
        pass
    for i in range(2, 8):  # 2, 3, 4, 5, 6, 7
        pass
    for i in range(0, 10, 2):  # 0, 2, 4, 6, 8（步长为 2）
        pass

    # ---- enumerate() — 带索引的遍历 ----
    # 类比 Java: for (int i = 0; i < list.size(); i++)
    fruits = ["apple", "banana", "cherry"]
    for i, fruit in enumerate(fruits):
        print(f"  {i}: {fruit}")

    # ---- zip() — 并行遍历两个列表 ----
    # 类比 Java: for (int i = 0; i < keys.size(); i++)
    keys = ["a", "b", "c"]
    values = [1, 2, 3]
    for k, v in zip(keys, values):
        print(f"  {k} -> {v}")

    # ---- sorted() / list.sort() ----
    # 类比 Java: Collections.sort(list) / Stream.sorted()
    nums = [3, 1, 4, 1, 5, 9]
    print(f"  sorted([3,1,4]) = {sorted(nums)}")  # 新列表 [1, 1, 3, 4, 5, 9]
    nums_copy = nums.copy()
    nums_copy.sort(reverse=True)  # 原地排序 [9, 5, 4, 3, 1, 1]
    print(f"  sort reverse = {nums_copy}")

    # ---- map() / filter() ----
    # 类比 Java: stream().map() / stream().filter()
    nums = [1, 2, 3, 4, 5]
    doubled = list(map(lambda x: x * 2, nums))  # [2, 4, 6, 8, 10]
    evens = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
    print(f"  map double = {doubled}")
    print(f"  filter even = {evens}")

    # ---- join() — 字符串拼接 ----
    # 类比 Java: String.join(",", list) / StringBuilder
    parts = ["a", "b", "c"]
    result = "-".join(parts)  # "a-b-c"
    print(f"  '-'.join(['a','b','c']) = '{result}'")

    # ---- split() — 字符串分割 ----
    # 类比 Java: string.split(regex)
    text = "a-b-c"
    parts = text.split("-")  # ["a", "b", "c"]
    print(f"  'a-b-c'.split('-') = {parts}")

    # ---- strip() — 去空白 ----
    # 类比 Java: string.trim()
    text = "  hello  "
    print(f"  '{text.strip()}'")  # "hello"

    # ---- isinstance() — 类型检查 ----
    # 类比 Java: obj instanceof String
    x = "hello"
    print(f"  isinstance('hello', str) = {isinstance(x, str)}")  # True

    # ---- getattr() / setattr() — 反射式访问属性 ----
    # 类比 Java: field.get(obj) / field.set(obj, val)
    class Person:
        name = "Alice"
        age = 25

    p = Person()
    print(f"  getattr(p, 'name') = {getattr(p, 'name')}")  # "Alice"
    print(f"  getattr(p, 'age', 0) = {getattr(p, 'age', 0)}")  # 25（默认值）

    print()


# ============================================================================
# 第 7 节：Python 常用第三方库简介（Java 程序员对照表）
# ============================================================================

def demo_common_libraries():
    """
    本项目中会用到的主要第三方库，及其 Java 等价物

    每个库都给出了最简单的使用示例。
    """

    # ------- 1. json — JSON 序列化/反序列化 -------
    # 类比 Java: Jackson / Gson / Fastjson
    import json

    data = {"name": "Alice", "age": 25, "hobbies": ["reading", "coding"]}

    # 对象 → JSON 字符串（序列化）
    # 类比 Java: jsonString = objectMapper.writeValueAsString(data);
    json_str = json.dumps(data, ensure_ascii=False)  # ensure_ascii=False 保证中文正常显示
    print(f"json.dumps: {json_str}")

    # JSON 字符串 → 对象（反序列化）
    # 类比 Java: Data obj = objectMapper.readValue(jsonString, Data.class);
    parsed = json.loads(json_str)
    print(f"json.loads: {parsed}")
    print()

    # ------- 2. re — 正则表达式 -------
    # 类比 Java: java.util.regex.Pattern / Matcher
    import re

    text = "A公司控股B公司，B公司控股C公司"

    # 查找所有 "X公司" 的模式
    # 类比 Java: Pattern.compile("[A-Z]公司").matcher(text).results()
    matches = re.findall(r'[A-Z]公司', text)
    print(f"re.findall: {matches}")  # ['A公司', 'B公司', 'C公司']

    # 搜索第一个匹配
    # 类比 Java: Pattern.compile("[A-Z]+公司").matcher(text).find()
    match = re.search(r'([A-Z]公司)控股([A-Z]公司)', text)
    if match:
        print(f"re.search group(0) = {match.group(0)}")  # 整个匹配
        print(f"re.search group(1) = {match.group(1)}")  # 第一个括号
        print(f"re.search group(2) = {match.group(2)}")  # 第二个括号
    print()

    # ------- 3. logging — 日志框架 -------
    # 类比 Java: SLF4J + Logback
    import logging

    # 配置日志（类比 logback.xml）
    logging.basicConfig(
        level=logging.INFO,  # DEBUG < INFO < WARNING < ERROR < CRITICAL
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)  # __name__ = 当前模块名（类比 Java 的 MyClass.class）

    logger.debug("这是调试信息")  # DEBUG 级别（开发时用）
    logger.info("这是普通信息")  # INFO 级别（正常运行）
    logger.warning("这是警告")  # WARNING 级别（潜在问题）
    logger.error("这是错误")  # ERROR 级别（出错了）
    print()

    # ------- 4. os / pathlib — 文件路径操作 -------
    # 类比 Java: java.nio.file.Paths / java.io.File
    import os
    from pathlib import Path

    # 获取当前工作目录
    cwd = os.getcwd()
    print(f"os.getcwd() = {cwd}")

    # 拼接路径（推荐 pathlib，类比 Java Paths.get()）
    p = Path("/home/user") / "documents" / "file.txt"
    print(f"pathlib: {p}")

    # 检查文件是否存在
    print(f"os.path.exists('/etc'): {os.path.exists('/etc')}")

    # 列出目录内容
    print(f"os.listdir('.'): {os.listdir('.')[:5]}...")  # 只显示前5个
    print()

    # ------- 5. dataclasses — 数据类 -------
    # 类比 Java: Lombok @Data / Java 14+ record
    from dataclasses import dataclass

    @dataclass
    class Employee:
        """员工数据类"""
        name: str  # 姓名
        department: str  # 部门
        salary: float = 0  # 薪资（有默认值）

    emp = Employee("张三", "技术部", 20000)
    print(f"dataclass: {emp}")  # 自动生成的 __repr__
    print(f"emp.name = {emp.name}")  # 属性访问
    print()

    # ------- 6. typing — 类型提示 -------
    # 类比 Java 的泛型，但 Python 是"软类型"，运行时不强制
    from typing import List, Dict, Optional, Tuple

    # List[str] ≈ List<String>
    names: List[str] = ["Alice", "Bob"]

    # Dict[str, int] ≈ Map<String, Integer>
    ages: Dict[str, int] = {"Alice": 25, "Bob": 30}

    # Optional[str] ≈ String | null
    maybe_name: Optional[str] = None

    # Tuple[int, str] ≈ 固定长度的元组
    point: Tuple[int, int] = (10, 20)

    print(f"typing: names={names}, ages={ages}")
    print()


# ============================================================================
# 第 8 节：完整实战 — 从文档到 Node 的全流程
# ============================================================================

def demo_full_pipeline():
    """
    完整 RAG 管线演示

    流程：
    1. 读取文档（Document）
    2. 选择切分策略（Splitter）
    3. 切分成节点（Node）
    4. 查看结果

    这相当于 Java 里的：
    Document doc = Files.readString(path);
    List<Node> nodes = splitter.split(doc);
    for (Node node : nodes) { ... }
    """
    from llama_index.core import Document
    from llama_index.core.node_parser import TokenTextSplitter

    # 第 1 步：模拟读取了一个文档
    # 实际项目中可能是：PDF / Word / HTML / 数据库 / Web 抓取
    document_content = """
    第一章 总则
    第一条 为规范公司管理，特制定本制度。
    第二条 本制度适用于全体员工。

    第二章 考勤管理
    第三条 员工每日上班时间为上午9:00至下午18:00。
    第四条 每周工作5天，周六周日休息。
    第五条 迟到早退超过30分钟按旷工半天处理。
    第六条 每月允许3次迟到豁免机会。

    第三章 薪酬福利
    第七条 员工月薪于次月15日发放。
    第八条 公司提供五险一金。
    第九条 年终奖金根据公司盈利情况和个人绩效评定。
    第十条 员工享有带薪年假、病假、婚假、产假等国家规定的假期。

    第四章 附则
    第十一条 本制度自发布之日起执行。
    第十二条 本制度由人力资源部负责解释。
    """

    # 创建 Document 对象
    # 类比 Java: new Document(content, metadata)
    doc = Document(
        text=document_content,
        metadata={
            "filename": "员工手册.txt",
            "category": "人力资源",
            "version": "2024.v1"
        }
    )

    print("=" * 60)
    print("第 1 步：加载文档")
    print("=" * 60)
    print(f"  文档标题: {doc.metadata.get('filename', 'N/A')}")
    print(f"  分类: {doc.metadata.get('category', 'N/A')}")
    print(f"  总字符数: {len(doc.text)}")
    print()

    # 第 2 步：选择切分策略
    print("=" * 60)
    print("第 2 步：使用 TokenTextSplitter 进行切分")
    print("=" * 60)
    print("  参数: chunk_size=128, chunk_overlap=16, separator=\\n")
    print()

    splitter = TokenTextSplitter(
        chunk_size=128,  # 每块最多 128 token（约 200~300 中文字符）
        chunk_overlap=16,  # 相邻块重叠 16 token（约 20~30 字符）
        separator="\n"  # 优先在换行处切分
    )

    # 第 3 步：执行切分
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"  原始文档: {len(doc.text)} 字符")
    print(f"  切分结果: {len(nodes)} 个节点")
    print()

    # 第 4 步：查看每个节点
    print("=" * 60)
    print("第 3 步：查看切分结果")
    print("=" * 60)

    for i, node in enumerate(nodes):
        print(f"\n--- 节点 {i + 1}/{len(nodes)} ---")
        print(f"  文本: {node.text.strip()}")
        print(f"  字符数: {len(node.text)}")
        print(f"  元数据: filename={node.metadata.get('filename')}, category={node.metadata.get('category')}")
        print(f"  所属文档ID: {node.ref_doc_id}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


# ============================================================================
# 第 9 节：5 种切片方法对比总结
# ============================================================================
"""
┌────────────────────┬──────────┬──────────┬──────────┬───────────────┐
│       方法          │  智能程度 │  速度    │  成本    │   适用场景     │
├────────────────────┼──────────┼──────────┼──────────┼───────────────┤
│ 手动切片            │  ★       │  极快    │  免费    │  学习原理      │
│ Token 切片          │  ★★★    │  快      │  免费    │  通用场景(最常用)│
│ 句子切片            │  ★★★★  │  快      │  免费    │  FAQ/对话体    │
│ 语义切片            │  ★★★★★ │  慢      │  有成本  │  高质量RAG     │
│ Markdown 切片       │  ★★★★  │  快      │  免费    │  技术文档/Wiki  │
└────────────────────┴──────────┴──────────┴──────────┴───────────────┘

【如何选择？】
  - 刚开始做 RAG？先用 Token 切片，最简单也最有效
  - 文档有明确段落/标题？用 Markdown 切片
  - 文档是对话/FAQ？用句子切片
  - 对检索质量要求极高、预算充足？用语义切片
  - 实际项目中经常混合使用多种策略
"""

# ============================================================================
# 入口点
# ============================================================================
if __name__ == "__main__":
    import sys

    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   RAG 学习笔记：Node（节点）与文本切片（Chunking）         ║
║   面向 Java 程序员的 Python 零基础入门                      ║
║                                                          ║
║   本节内容：                                              ║
║   1. Python 基础速成                                      ║
║   2. Python 标准库 vs Java 包对照                         ║
║   3. Document / Node / TextNode 概念                      ║
║   4. 5 种切片方法详解                                     ║
║   5. Node 属性详解                                        ║
║   6. Python 常用内建函数                                  ║
║   7. 常用第三方库简介                                     ║
║   8. 完整实战：从文档到 Node                               ║
║   9. 5 种方法对比总结                                     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    # 如果某些演示需要联网调用 API（语义切片、Markdown 切片），
    # 可以注释掉对应的行跳过

    print("━━━ 第 1 节：手动切片 ━━━")
    demo_manual_split()

    print("━━━ 第 2 节：Token 切片 ━━━")
    demo_token_split()

    print("━━━ 第 3 节：句子切片 ━━━")
    demo_sentence_split()

    print("━━━ 第 4 节：语义切片 ━━━")
    try:
        demo_semantic_split()
    except Exception as e:
        print(f"  跳过（需要联网调用 Embedding API）: {e}")

    print("━━━ 第 5 节：Markdown 切片 ━━━")
    try:
        demo_markdown_split()
    except Exception as e:
        print(f"  跳过: {e}")

    print("━━━ 第 6 节：Node 属性 ━━━")
    demo_node_attributes()

    print("━━━ 第 7 节：Python 内建函数 ━━━")
    demo_python_builtins()

    print("━━━ 第 8 节：常用库演示 ━━━")
    demo_common_libraries()

    print("━━━ 第 9 节：完整实战 ━━━")
    demo_full_pipeline()

    print("\n🎉 全部演示完成！")
    print("   建议下一步：阅读 week03/code/p22-node.ipynb 和 p27-切片.ipynb 的 Jupyter notebook")
