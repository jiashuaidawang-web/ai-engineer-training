"""
02_document_ingestion.py — 文档加载与解析

这一课深入学习 LlamaIndex 的文档加载能力：
1. 支持的文件格式（PDF, MD, TXT, DOCX, PPTX, XLSX, HTML, 图片）
2. 自定义解析器
3. 元数据提取
4. 增量加载

【Java 程序员速查】
  LlamaIndex 的文档加载 = Java 的文件读取 + 解析
  但 LlamaIndex 做了很多适配：自动识别文件格式、提取元数据、统一接口
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,    # 简单目录加载器（最常用）
    Settings,                # 全局配置
    Document,                # 文档对象
)
from llama_index.readers.file import (  # 文件读取器模块
    PDFReader,               # PDF 解析器
    MarkdownReader,          # Markdown 解析器
    TXTReader,               # 纯文本解析器
    DocxReader,              # Word 文档解析器
    PptxReader,              # PowerPoint 解析器
    ExcelReader,             # Excel 解析器
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置 LLM
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)
Settings.embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

print("=" * 60)
print("  文档加载与解析 — Document Ingestion")
print("=" * 60)


# ============================================================
# 1. SimpleDirectoryReader — 最简单的方式
# ============================================================

def demo_simple_directory_reader():
    """
    SimpleDirectoryReader 是 LlamaIndex 最推荐的文档加载方式

    它会自动：
    1. 扫描指定目录下的所有文件
    2. 根据文件扩展名选择对应的解析器
    3. 提取元数据（文件名、路径、修改时间等）
    4. 返回统一的 Document 对象列表

    类比 Java:
      // 你需要手动写很多 if-else 判断文件类型
      if (file.endsWith(".pdf")) { pdfReader.parse(file); }
      else if (file.endsWith(".md")) { mdReader.parse(file); }
      // ...

      // SimpleDirectoryReader 帮你自动处理所有类型
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "llamaindex-demo", "data")

    if not os.path.exists(data_dir):
        print(f"\n[错误] 数据目录不存在: {data_dir}")
        print("请先将测试文档放入 llamaindex-demo/data/ 目录")
        return

    print(f"\n>>> 使用 SimpleDirectoryReader 加载目录: {data_dir}")

    # --- 【Python 语法】SimpleDirectoryReader(data_dir).load_data() ---
    # 一步完成：扫描目录 → 识别文件 → 解析 → 返回 Document 列表
    # 类比 Java: new SimpleDirectoryReader(dataDir).loadDocuments()
    documents = SimpleDirectoryReader(data_dir).load_data()

    print(f"[结果] 共加载 {len(documents)} 个文档")

    # --- 【Python 语法】Document 对象的属性 ---
    # doc.id_ — 文档唯一 ID
    # doc.text — 文档文本内容
    # doc.metadata — 元数据字典（文件名、路径、作者等）
    # doc.excluded_llm_metadata_keys — 不发送给 LLM 的元数据
    # doc.excluded_embed_metadata_keys — 不用于 embedding 的元数据
    for i, doc in enumerate(documents):
        print(f"\n  文档 {i+1}:")
        print(f"    ID: {doc.id_}")
        print(f"    文件名: {doc.metadata.get('file_name', 'N/A')}")
        print(f"    文件路径: {doc.metadata.get('file_path', 'N/A')}")
        print(f"    内容长度: {len(doc.text)} 字符")
        print(f"    内容预览: {doc.text[:100]}...")

    return documents


# ============================================================
# 2. 自定义解析器 — 针对特定格式
# ============================================================

def demo_custom_parsers():
    """
    对于特殊格式的文件，可以使用自定义解析器

    每种解析器都有相同的接口：
      parser = XxxReader()
      documents = parser.load_data(file_path)

    类比 Java:
      PdfParser parser = new PdfParser();
      List<Document> docs = parser.parse(file);

      MarkdownParser parser = new MarkdownParser();
      List<Document> docs = parser.parse(file);
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "llamaindex-demo", "data")

    print(f"\n>>> 使用自定义解析器...")

    # --- 示例：加载单个 PDF 文件 ---
    # 找到第一个 PDF 文件
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]

    if pdf_files:
        pdf_path = os.path.join(data_dir, pdf_files[0])
        print(f"\n  [PDF] 加载文件: {pdf_files[0]}")

        # --- 【Python 语法】PDFReader ---
        # 注意：PDFReader 需要安装 pypdf2
        # pip install pypdf2
        try:
            from llama_index.readers.file import PDFReader
            pdf_parser = PDFReader()  # 创建 PDF 解析器实例
            # load_data 返回 Document 列表
            docs = pdf_parser.load_data(pdf_path)
            print(f"  [PDF] 解析出 {len(docs)} 页")
            for i, doc in enumerate(docs):
                print(f"    第 {i+1} 页: {len(doc.text)} 字符")
        except ImportError:
            print("  [PDF] 跳过（需要安装 pypdf2: pip install pypdf2）")

    # --- 示例：加载 Markdown 文件 ---
    md_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
    if md_files:
        md_path = os.path.join(data_dir, md_files[0])
        print(f"\n  [MD] 加载文件: {md_files[0]}")

        try:
            from llama_index.readers.file import MarkdownReader
            md_parser = MarkdownReader()  # 创建 Markdown 解析器
            docs = md_parser.load_data(md_path)
            print(f"  [MD] 解析出 {len(docs)} 个文档块")
            for i, doc in enumerate(docs):
                print(f"    块 {i+1}: {len(doc.text)} 字符")
        except ImportError:
            print("  [MD] 跳过（需要安装 markdown 解析器）")

    # --- 示例：加载 TXT 文件 ---
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if txt_files:
        txt_path = os.path.join(data_dir, txt_files[0])
        print(f"\n  [TXT] 加载文件: {txt_files[0]}")

        try:
            from llama_index.readers.file import TXTReader
            txt_parser = TXTReader()  # 创建文本解析器
            docs = txt_parser.load_data(txt_path)
            print(f"  [TXT] 解析出 {len(docs)} 个文档")
            for i, doc in enumerate(docs):
                print(f"    文档 {i+1}: {len(doc.text)} 字符")
        except ImportError:
            print("  [TXT] 跳过")


# ============================================================
# 3. 元数据提取
# ============================================================

def demo_metadata_extraction():
    """
    元数据（Metadata）是文档的「描述信息」

    每个 Document 对象都带有 metadata 字典，包含：
    - file_name: 文件名
    - file_path: 文件路径
    - file_size: 文件大小（字节）
    - last_modified: 最后修改时间
    - 自定义字段（你可以添加任何需要的元数据）

    元数据的用途：
    1. 过滤：搜索时只查特定来源的文档
    2. 追溯：回答时引用来源
    3. 权限：根据元数据控制访问

    类比 Java:
      Document doc = new Document("内容");
      doc.setMetadata("fileName", "test.pdf");
      doc.setMetadata("author", "张三");
      Map<String, Object> metadata = doc.getMetadata();
    """
    print(f"\n>>> 元数据提取示例")

    # --- 【Python 语法】创建 Document 对象 ---
    # Document 是 LlamaIndex 的核心数据类
    doc = Document(
        text="这是一段测试文本",
        metadata={  # 自定义元数据
            "source": "manual",  # 来源标记
            "author": "学习者",   # 作者
            "tags": ["测试", "学习"],  # 标签列表
        },
    )

    print(f"  文档内容: {doc.text}")
    print(f"  元数据: {doc.metadata}")
    print(f"  作者: {doc.metadata.get('author')}")
    print(f"  标签: {', '.join(doc.metadata.get('tags', []))}")


# ============================================================
# 4. 增量加载
# ============================================================

def demo_incremental_loading():
    """
    增量加载 = 只加载新增或修改过的文件

    使用 filename_to_doc_id 映射，记录哪些文件已经加载过。
    下次运行时，只加载新文件。

    类比 Java:
      // 维护一个已加载文件的集合
      Set<String> loadedFiles = new HashSet<>();
      // 扫描目录
      for (File file : directory.listFiles()) {
          if (!loadedFiles.contains(file.getName())) {
              loadDocument(file);
              loadedFiles.add(file.getName());
          }
      }
    """
    print(f"\n>>> 增量加载示例")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "llamaindex-demo", "data")

    if not os.path.exists(data_dir):
        print("  [跳过] 数据目录不存在")
        return

    # --- 【Python 语法】filename_to_doc_id ---
    # 这是一个字典，记录文件名 → 文档 ID 的映射
    # 用于增量加载时判断文件是否已处理
    filename_to_doc_id = {}

    # 模拟：只加载 .txt 文件
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    if txt_files:
        print(f"  找到 {len(txt_files)} 个 TXT 文件")
        for fname in txt_files:
            file_path = os.path.join(data_dir, fname)
            # 检查是否已加载
            if fname not in filename_to_doc_id:
                print(f"    [新增] {fname}")
                # 这里可以加载文件并记录
                # filename_to_doc_id[fname] = generate_unique_id()
            else:
                print(f"    [已存在] {fname}")
    else:
        print("  没有找到 TXT 文件")


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数：依次运行所有文档加载演示
    """
    print("\n[1] SimpleDirectoryReader")
    demo_simple_directory_reader()

    print("\n[2] 自定义解析器")
    demo_custom_parsers()

    print("\n[3] 元数据提取")
    demo_metadata_extraction()

    print("\n[4] 增量加载")
    demo_incremental_loading()

    print("\n" + "=" * 60)
    print("  OK 文档加载与解析完成！")
    print("=" * 60)
    print("""
下一步：
  - 将你的文档放入 llamaindex-demo/data/ 目录
  - 支持 PDF, MD, TXT, DOCX, PPTX, XLSX, HTML 等格式
  - 运行后会看到每个文件的解析结果
    """)


if __name__ == "__main__":
    main()
