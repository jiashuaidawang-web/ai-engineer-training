"""
===============================================================================
 第 3 课：Reader / 文档加载 — 从文件读取 Document
===============================================================================

【这一课学什么？】
  第 1 课我们手动创建了 Document，第 2 课配置了 Settings。
  但现实中，文档是从硬盘上的 PDF、Word、TXT、Excel 等文件中来的。
  Reader 就是负责把这些文件读取并转换成 Document 对象的组件。

【类比 Java】
  Reader ≈ Apache Tika / Apache POI / iText
  - Tika 自动识别文件类型并提取文本
  - POI 专门读 Excel/Word
  - iText 专门读 PDF
  LlamaIndex 的 Reader 就是这些工具的封装，统一输出 Document 格式。

【核心概念】
  Reader 的工作流程：
    文件(.pdf/.txt/.docx) → Reader 解析 → Document(text + metadata)

  常用 Reader：
    - SimpleDirectoryReader  → 读取整个文件夹（最常用！）
    - SimpleFileReader      → 读取单个文件
    - PDFReader             → 专门处理 PDF
    - DocxReader            → 专门处理 Word
    - UnstructuredReader    → 处理复杂格式（HTML、邮件等）

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/03_Reader文档加载.py

【前置知识】
  - 第 1 课：Node / 文本切片
  - 第 2 课：Settings / 全局配置
"""

import os
from pathlib import Path
from llama_index.core import Settings


# ============================================================================
# 第 1 节：SimpleDirectoryReader — 最核心的 Reader
# ============================================================================

def demo_simple_directory_reader():
    """
    SimpleDirectoryReader 是 LlamaIndex 最常用的文档加载器

    它能：
    - 递归扫描一个文件夹下的所有文件
    - 自动识别文件类型（.txt, .pdf, .docx, .xlsx, .csv, .json 等）
    - 提取文本内容和元数据（文件名、修改时间等）
    - 返回 Document 列表

    类比 Java：
      // 伪代码
      List<Document> docs = new SimpleDirectoryReader("./documents/")
          .load()
          .stream()
          .map(file -> new Document(file.getName(), readFileContent(file)))
          .collect(toList());
    """
    print("=" * 60)
    print("【SimpleDirectoryReader 演示】")
    print("=" * 60)

    from llama_index.core.readers.simple_file_based import SimpleDirectoryReader

    # 创建一个测试目录和文件
    test_dir = Path("./test_docs")
    test_dir.mkdir(exist_ok=True)

    # 写入几个测试文件
    (test_dir / "考勤制度.txt").write_text(
        "公司实行每日八小时工作制。\n"
        "上午工作时间为九时至十二时。\n"
        "下午工作时间为十三时三十分至十七时三十分。\n"
        "员工每周享有两天休息日，通常为周六和周日。",
        encoding="utf-8"
    )

    (test_dir / "休假制度.txt").write_text(
        "员工可享受以下假期：法定节假日、带薪年休假、病假、事假、婚假、产假等。\n"
        "工作满1年不满10年的，年休假5天。\n"
        "满10年不满20年的，年休假10天。\n"
        "满20年及以上的，年休假15天。",
        encoding="utf-8"
    )

    (test_dir / "薪酬制度.txt").write_text(
        "员工月薪于次月15日发放。\n"
        "公司提供五险一金。\n"
        "年终奖金根据公司盈利情况和个人绩效评定。",
        encoding="utf-8"
    )

    print(f"\n  已创建测试目录: {test_dir}")
    print(f"  包含 {len(list(test_dir.glob('*')))} 个文件\n")

    # 使用 SimpleDirectoryReader 读取整个文件夹
    # 参数说明：
    #   input_dir   → 要读取的目录路径
    #   required_exts → 只读取特定扩展名的文件（可选）
    #   file_mode   → 文件读取模式（"text"=逐行读, "binary"=二进制读）
    #   recursive  → 是否递归读取子目录（默认 True）
    reader = SimpleDirectoryReader(
        input_dir=str(test_dir),
        required_exts=[".txt"],       # 只读取 .txt 文件
        recursive=True,               # 递归子目录
        file_mode="text",             # 以文本模式读取
    )

    # load_data() 是核心方法：读取所有文件并返回 Document 列表
    # 类比 Java: List<Document> docs = reader.load_data();
    documents = reader.load_data()

    print(f"  共加载 {len(documents)} 个 Document:\n")

    for i, doc in enumerate(documents):
        print(f"  --- Document {i + 1} ---")
        print(f"    文本长度: {len(doc.text)} 字符")
        print(f"    文本前 50 字: {doc.text[:50]}...")
        print(f"    元数据: {dict(doc.metadata)}")
        print()

    # 清理测试文件
    for f in test_dir.glob("*"):
        f.unlink()
    test_dir.rmdir()
    print(f"  ✓ 已清理测试目录\n")


# ============================================================================
# 第 2 节：SimpleFileReader — 读取单个文件
# ============================================================================

def demo_simple_file_reader():
    """
    SimpleFileReader 用于读取单个文件

    当你只需要处理一个文件时，比 SimpleDirectoryReader 更轻量。
    """
    print("=" * 60)
    print("【SimpleFileReader 演示】")
    print("=" * 60)

    from llama_index.core.readers.simple_file_based import SimpleFileReader

    # 创建一个临时测试文件
    test_file = Path("./test_single.txt")
    test_file.write_text(
        "这是一篇关于 RAG 技术的短文。\n"
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。\n"
        "它先从知识库中检索相关信息，然后将信息作为上下文提供给 LLM 生成回答。",
        encoding="utf-8"
    )

    print(f"\n  创建测试文件: {test_file}\n")

    # 读取单个文件
    reader = SimpleFileReader()
    # load_data() 接受文件路径列表
    # 返回值是 (Document, extra_info) 的元组列表
    documents = reader.load_data([str(test_file)])

    print(f"  加载了 {len(documents)} 个 Document:")
    for doc in documents:
        print(f"    文本: {doc.text[:60]}...")
        print(f"    元数据: {dict(doc.metadata)}")

    # 清理
    test_file.unlink()
    print(f"\n  ✓ 已清理测试文件")


# ============================================================================
# 第 3 节：Reader 的元数据（Metadata）
# ============================================================================

def demo_metadata():
    """
    Reader 不仅提取文本，还会自动提取元数据

    元数据就像是文件的"身份证"，包含：
    - 文件名（source_file）
    - 文件路径（file_path）
    - 文件大小（file_size）
    - 修改时间（file_mdate）
    - 自定义元数据（你可以额外添加）

    类比 Java：
      // 读取文件时同时提取元数据
      Document doc = new Document(content, Map.of(
          "filename", file.getName(),
          "size", file.length(),
          "lastModified", file.lastModified()
      ));
    """
    print("=" * 60)
    print("【Reader 的元数据提取】")
    print("=" * 60)

    from llama_index.core.readers.simple_file_based import SimpleDirectoryReader

    # 创建测试文件
    test_dir = Path("./test_meta")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "test.txt").write_text("测试内容", encoding="utf-8")

    # 方式 1：让 Reader 自动提取元数据
    reader1 = SimpleDirectoryReader(input_dir=str(test_dir))
    docs1 = reader1.load_data()
    doc1 = docs1[0]

    print("\n  --- 方式 1：自动提取的元数据 ---")
    print(f"    source_file: {doc1.metadata.get('source_file')}")
    print(f"    file_path:   {doc1.metadata.get('file_path')}")
    print(f"    file_size:   {doc1.metadata.get('file_size')}")

    # 方式 2：添加自定义元数据（类似 Java 的 map.put()）
    reader2 = SimpleDirectoryReader(
        input_dir=str(test_dir),
        required_metadata={       # 给所有文档添加统一的额外元数据
            "category": "人力资源",
            "version": "2024.v1",
            "author": "系统管理员"
        }
    )
    docs2 = reader2.load_data()
    doc2 = docs2[0]

    print("\n  --- 方式 2：添加自定义元数据 ---")
    for key, value in doc2.metadata.items():
        print(f"    {key}: {value}")

    # 清理
    (test_dir / "test.txt").unlink()
    test_dir.rmdir()


# ============================================================================
# 第 4 节：Reader 的常见类型对照表
# ============================================================================

def demo_reader_types():
    """
    LlamaIndex 提供了多种 Reader，每种针对不同的数据源

    这个表格帮助你快速选择合适的 Reader。
    """
    print("=" * 60)
    print("【Reader 类型对照表】")
    print("=" * 60)

    print("""
  ┌────────────────────────────┬──────────────────────┬─────────────────────────────┐
  │ Reader 类名                │ 输入格式             │ 类比 Java                     │
  ├────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ SimpleDirectoryReader      │ 文件夹（多种格式）    │ FileUtils.listFiles() + Tika │
  │ SimpleFileReader           │ 单个文本文件          │ Files.readString(path)       │
  │ PDFReader                  │ PDF 文件             │ iText / PDFBox               │
  │ DocxReader                 │ Word .docx           │ Apache POI                   │
  │ UnstructuredReader         │ HTML/邮件/复杂格式    │ Jsoup + 自定义解析           │
  │ URLReader                  │ URL 网页             │ HttpClient + Jsoup           │
  │ JSONReader                 │ JSON 文件            │ Jackson / Gson               │
  │ CSVReader                  │ CSV 文件             │ OpenCSV / Apache Commons CSV │
  │ XLSXReader                 │ Excel .xlsx          │ Apache POI / jxl             │
  │ SlackReader                │ Slack 消息           │ Slack Java API               │
  │ NotionReader               │ Notion 页面          │ Notion API                   │
  │ GitHubReader               │ GitHub 仓库          │ GitHub Java API              │
  │ VectorStoreRetriever       │ 已有 VectorStore      │ 从数据库加载                   │
  └────────────────────────────┴──────────────────────┴─────────────────────────────┘

  最常用的是 SimpleDirectoryReader（80% 的场景都用它）。
  其他 Reader 需要额外安装对应的包，如：
    pip install llama-index-readers-file    # 包含 PDF/DOCX/CSV 等
    pip install llama-index-readers-web     # 包含 URLReader
    pip install llama-index-readers-notion  # Notion 专用
    """)


# ============================================================================
# 第 5 节：Reader + NodeParser 的组合使用
# ============================================================================

def demo_reader_plus_parser():
    """
    演示 Reader 和 NodeParser 如何配合工作

    这是 RAG 管线的前两步：
      文件 → Reader → Document → NodeParser → Node

    类比 Java：
      // 读取文件（Reader）
      String content = Files.readString(path);
      Document doc = new Document(content, metadata);

      // 切分文档（NodeParser）
      List<Node> nodes = splitter.getNodesFromDocument(doc);
    """
    print("=" * 60)
    print("【Reader + NodeParser 组合使用】")
    print("=" * 60)

    from llama_index.core.readers.simple_file_based import SimpleDirectoryReader
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.core import Document

    # 创建测试文件
    test_dir = Path("./test_combo")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "公司制度.txt").write_text(
        "第一章 总则\n"
        "为规范公司管理，特制定本制度。\n"
        "本制度适用于全体员工。\n\n"
        "第二章 考勤管理\n"
        "员工每日上班时间为上午9:00至下午18:00。\n"
        "每周工作5天，周六周日休息。\n"
        "迟到早退超过30分钟按旷工半天处理。\n"
        "每月允许3次迟到豁免机会。\n\n"
        "第三章 薪酬福利\n"
        "员工月薪于次月15日发放。\n"
        "公司提供五险一金。\n"
        "年终奖金根据公司盈利情况和个人绩效评定。",
        encoding="utf-8"
    )

    print("\n  --- 第 1 步：Reader 读取文件 → Document ---")
    reader = SimpleDirectoryReader(input_dir=str(test_dir), required_exts=[".txt"])
    documents = reader.load_data()
    print(f"    读取了 {len(documents)} 个 Document")
    print(f"    第一个 Document 的文本长度: {len(documents[0].text)} 字符")
    print(f"    元数据: {dict(documents[0].metadata)}")

    print("\n  --- 第 2 步：NodeParser 切分 Document → Node ---")
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"    切分成 {len(nodes)} 个 Node")

    for i, node in enumerate(nodes):
        print(f"\n    Node {i + 1}:")
        print(f"      文本: {node.text.strip()[:60]}...")
        print(f"      字符数: {len(node.text)}")

    # 清理
    (test_dir / "公司制度.txt").unlink()
    test_dir.rmdir()
    print(f"\n  ✓ 组合流程演示完成！")


# ============================================================================
# 第 6 节：Reader 的高级用法 — 自定义 Reader
# ============================================================================

def demo_custom_reader():
    """
    演示如何编写自定义 Reader

    当 LlamaIndex 内置的 Reader 无法满足需求时，
    你可以继承 BaseReader 来实现自己的 Reader。

    类比 Java：
      // 实现一个接口
      public class MyReader implements Reader {
          @Override
          public List<Document> load(String path) {
              // 自定义解析逻辑
          }
      }
    """
    print("=" * 60)
    print("【自定义 Reader 演示】")
    print("=" * 60)

    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document

    # 定义一个自定义 Reader 类
    # 类比 Java: public class LineDelimitedReader implements BaseReader
    class LineDelimitedReader(BaseReader):
        """
        逐行读取文本文件的自定义 Reader

        每一行被视为一个独立的 Document。
        适用于日志文件、逐行记录的数据等场景。
        """

        def __init__(self, delimiter="\n"):
            """
            构造函数

            参数：
              delimiter → 行分隔符，默认换行符
            类比 Java: public LineDelimitedReader(String delimiter)
            """
            super().__init__()
            self.delimiter = delimiter

        def load_data(self, lines):
            """
            加载数据的核心方法

            参数：
              lines → 文本行列表（List[str]）

            返回：
              Document 列表

            类比 Java:
              public List<Document> loadData(List<String> lines) {
                  return lines.stream()
                      .map(line -> new Document(line))
                      .collect(toList());
              }
            """
            # 将每一行包装成一个 Document
            # 注意：返回的是一个列表，里面只有一个 Document 对象
            # 这个 Document 的 text 属性是完整的 lines 列表转换成的字符串
            docs = [
                Document(
                    text=line,  # 每一行作为一个 Document 的文本
                    metadata={"line_number": idx + 1}  # 记录行号
                )
                for idx, line in enumerate(lines)
                if line.strip()  # 跳过空行
            ]
            return docs

    # 使用自定义 Reader
    print("\n  创建测试数据...")
    sample_lines = [
        "张三,技术部,20000",
        "李四,市场部,18000",
        "王五,人力资源部,16000",
        "",  # 空行（会被跳过）
        "赵六,财务部,22000",
    ]

    print("  使用自定义 Reader 加载数据...")
    custom_reader = LineDelimitedReader(delimiter="\n")
    documents = custom_reader.load_data(sample_lines)

    print(f"\n  加载了 {len(documents)} 个 Document：")
    for i, doc in enumerate(documents):
        print(f"    Doc {i + 1}: '{doc.text}' | 行号: {doc.metadata['line_number']}")

    print("\n  💡 自定义 Reader 让你可以灵活处理任何格式的数据！")


# ============================================================================
# 第 7 节：Reader 与 Java 文件读取的完整对照
# ============================================================================

def demo_java_comparison():
    """
    用 Java 的文件读取方式来类比 Python 的 Reader
    """
    print("=" * 60)
    print("【Reader vs Java 文件读取对照表】")
    print("=" * 60)

    print("""
  Java 代码                                  Python / LlamaIndex
  ─────────────────────────────────────      ─────────────────────────────────────
                                                                              ①
  // 读取整个文件为字符串                      from llama_index.core.readers.\\
  String content =                            simple_file_based import \\
      Files.readString(path);                  SimpleDirectoryReader

  // 读取文件夹下所有文件                      reader = SimpleDirectoryReader(
  Files.walk(path)                             input_dir="./docs"
      .filter(Files::isRegularFile)            )
      .forEach(file -> {                       docs = reader.load_data()
          String text = Files.readString(file);
          Document doc = new Document(text);
      });

  // 逐行读取                                   reader = SimpleFileReader()
  Files.lines(path)                            docs = reader.load_data(["file.txt"])
      .forEach(line -> { ... });

  // 解析 PDF                                   // 需要额外安装
  PDDocument doc = PDFParser.parse(file);      from llama_index.readers.pdf \\
  String text = doc.getText();                 import PDFReader
                                              reader = PDFReader()

  // 解析 Word                                  from llama_index.readers.docx \\
  XWPFDocument doc = new                      import DocxReader
      XWPFDocument(inputStream);              reader = DocxReader()

  // 解析 Excel                                 reader = XLSXReader()
  Workbook wb = new XSSFWorkbook(file);
  Sheet sheet = wb.getSheetAt(0);
    """)


# ============================================================================
# 第 8 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 3 课总结                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. SimpleDirectoryReader 是最常用的 Reader，一次性读取整个文件夹         │
│  2. Reader 的输出是 Document 列表（含文本 + 元数据）                      │
│  3. 元数据很重要：文件名、路径、自定义标签等，检索时用于过滤               │
│  4. 可以自定义 Reader 来处理特殊格式                                     │
│  5. 工作流：文件 → Reader → Document → NodeParser → Node                │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 读取整个文件夹（最常用）                                                │
│  from llama_index.core.readers.simple_file_based import \\              │
│      SimpleDirectoryReader                                               │
│  reader = SimpleDirectoryReader(input_dir="./docs")                      │
│  documents = reader.load_data()                                          │
│                                                                         │
│  # 添加自定义元数据                                                      │
│  reader = SimpleDirectoryReader(                                          │
│      input_dir="./docs",                                                 │
│      required_metadata={"category": "HR"}                               │
│  )                                                                      │
│                                                                         │
│  # 读取单个文件                                                          │
│  from llama_index.core.readers.simple_file_based import SimpleFileReader│
│  reader = SimpleFileReader()                                            │
│  documents = reader.load_data(["path/to/file.txt"])                      │
│                                                                         │
│  【下一课预告】                                                           │
│  第 4 课：Vector Store / 向量存储 — FAISS、Chroma 等                      │
│  类比 Java：JDBC / MongoDB Driver                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   第 3 课：Reader / 文档加载 — 从文件读取 Document        ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. SimpleDirectoryReader — 读取整个文件夹              ║
║   2. SimpleFileReader — 读取单个文件                    ║
║   3. Reader 的元数据（Metadata）                         ║
║   4. Reader 类型对照表                                   ║
║   5. Reader + NodeParser 组合使用                        ║
║   6. 自定义 Reader                                      ║
║   7. Reader vs Java 文件读取对照                         ║
║   8. 总结                                               ║
║                                                          ║
║   前置知识：第 1 课 Node / 第 2 课 Settings               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：SimpleDirectoryReader ━━━")
    demo_simple_directory_reader()

    print("\n━━━ 第 2 节：SimpleFileReader ━━━")
    demo_simple_file_reader()

    print("\n━━━ 第 3 节：Reader 的元数据 ━━━")
    demo_metadata()

    print("\n━━━ 第 4 节：Reader 类型对照表 ━━━")
    demo_reader_types()

    print("\n━━━ 第 5 节：Reader + NodeParser 组合 ━━━")
    demo_reader_plus_parser()

    print("\n━━━ 第 6 节：自定义 Reader ━━━")
    demo_custom_reader()

    print("\n━━━ 第 7 节：Reader vs Java 对照 ━━━")
    demo_java_comparison()

    print("\n🎉 第 3 课完成！")
    print("   建议下一步：阅读 week03/code/p25-reader.ipynb")
