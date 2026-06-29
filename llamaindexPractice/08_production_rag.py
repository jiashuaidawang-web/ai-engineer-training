"""
08_production_rag.py — 生产级 RAG 系统

这一课学习生产环境中的 RAG 最佳实践：
1. 混合搜索（向量 + 关键词）
2. 重排序（Reranking）
3. 缓存机制
4. 评估框架
5. 部署为 API 服务

【Java 程序员速查】
  生产级 RAG = 鲁棒性 + 可观测性 + 可维护性
  类比 Java 的生产级应用：
    - 日志（SLF4J/Logback）
    - 监控（Micrometer/Prometheus）
    - 缓存（Redis/Caffeine）
    - 限流（Sentinel/Hystrix）
"""

import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle

# 配置 LLM
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

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
print("  生产级 RAG 系统")
print("=" * 60)


# ============================================================
# 1. 混合搜索（Hybrid Search）
# ============================================================

class HybridSearchRetriever:
    """
    混合搜索检索器：向量搜索 + 关键词搜索

    为什么需要混合搜索？
    - 向量搜索：擅长语义理解（"猫" 能找到 "猫咪"）
    - 关键词搜索：擅长精确匹配（"Milvus" 必须找到含 "Milvus" 的文档）
    - 混合搜索：两者互补，覆盖更多场景

    类比 Java:
      // 向量搜索
      List<Node> vectorResults = vectorSearch(query);
      // 关键词搜索
      List<Node> keywordResults = keywordSearch(query);
      // 融合结果
      List<Node> hybridResults = fuse(vectorResults, keywordResults);
    """

    def __init__(self, vector_index, keyword_index, alpha=0.7):
        """
        初始化混合搜索检索器

        参数：
          vector_index: 向量索引
          keyword_index: 关键词索引
          alpha: 向量搜索的权重（0-1），1-alpha 是关键词搜索的权重
        """
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.alpha = alpha  # 向量搜索权重

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        执行混合搜索

        步骤：
        1. 分别执行向量搜索和关键词搜索
        2. 对结果进行归一化和融合
        3. 返回融合后的 Top-K 结果

        参数：
          query: 查询文本
          top_k: 返回结果数量

        返回：
          List[NodeWithScore] — 排序后的节点列表
        """
        # --- 【Python 语法】向量搜索 ---
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=top_k)
        vector_nodes = vector_retriever.retrieve(query)

        # --- 【Python 语法】关键词搜索 ---
        keyword_retriever = self.keyword_index.as_retriever()
        keyword_nodes = keyword_retriever.retrieve(query)

        # --- 【Python 语法】结果融合 ---
        # 简单加权融合：score = alpha * vector_score + (1-alpha) * keyword_score
        fused_scores = {}
        for node in vector_nodes:
            fused_scores[node.node.node_id] = node.score * self.alpha
        for node in keyword_nodes:
            node_id = node.node.node_id
            if node_id in fused_scores:
                fused_scores[node_id] += node.score * (1 - self.alpha)
            else:
                fused_scores[node_id] = node.score * (1 - self.alpha)

        # 按融合分数排序
        sorted_nodes = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            NodeWithScore(node=self._find_node(node_id, vector_nodes + keyword_nodes),
                          score=score)
            for node_id, score in sorted_nodes[:top_k]
        ]

    def _find_node(self, node_id: str, nodes: List[NodeWithScore]) -> NodeWithScore:
        """根据 ID 查找节点"""
        for node in nodes:
            if node.node.node_id == node_id:
                return node
        return None


# ============================================================
# 2. 缓存机制
# ============================================================

class QueryCache:
    """
    查询缓存：避免重复查询相同问题

    为什么需要缓存？
    - 减少 API 调用次数（省钱）
    - 加快响应速度（提速）
    - 保证回答一致性（相同问题永远得到相同回答）

    类比 Java:
      // Guava Cache
      Cache<String, String> cache = CacheBuilder.newBuilder()
          .maximumSize(1000)
          .expireAfterWrite(1, TimeUnit.HOURS)
          .build();

      String answer = cache.getIfPresent(query);
      if (answer == null) {
          answer = llm.generate(query);
          cache.put(query, answer);
      }
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化缓存

        参数：
          max_size: 最大缓存条目数
          ttl_seconds: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, dict] = {}  # {查询: {"回答": ..., "时间戳": ...}}

    def get(self, query: str) -> Optional[str]:
        """
        从缓存获取回答

        如果查询在缓存中且未过期，返回缓存的回答
        否则返回 None

        参数：
          query: 查询文本

        返回：
          str | None — 缓存的回答或 None
        """
        if query in self.cache:
            cached = self.cache[query]
            # 检查是否过期
            elapsed = time.time() - cached['timestamp']
            if elapsed < self.ttl_seconds:
                return cached['answer']
            else:
                del self.cache[query]  # 删除过期条目
        return None

    def put(self, query: str, answer: str):
        """
        将查询-回答对存入缓存

        如果缓存已满，删除最老的条目

        参数：
          query: 查询文本
          answer: LLM 的回答
        """
        # 缓存满时删除最老的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(),
                             key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[query] = {
            'answer': answer,
            'timestamp': time.time(),
        }

    def clear(self):
        """清空缓存"""
        self.cache.clear()


# ============================================================
# 3. 评估框架
# ============================================================

class RagEvaluator:
    """
    RAG 系统评估器

    评估指标：
    1. 相关性：检索的文档是否与问题相关
    2. 准确性：回答是否准确
    3. 完整性：回答是否覆盖了问题的所有方面
    4. 时效性：回答是否使用了最新的文档

    类比 Java:
      // 单元测试评估
      @Test
      void testRagAccuracy() {
          String answer = ragSystem.query("什么是 RAG?");
          assertTrue(answer.contains("检索增强生成"));
          assertEquals(4.5, evaluateQuality(answer), 0.1);
      }
    """

    def __init__(self):
        """初始化评估器"""
        self.evaluation_results = []

    def evaluate(self, query: str, expected_keywords: List[str],
                 actual_answer: str) -> dict:
        """
        评估回答质量

        简单评估：检查回答是否包含预期的关键词

        参数：
          query: 原始问题
          expected_keywords: 预期回答中应包含的关键词
          actual_answer: 实际回答

        返回：
          dict — 评估结果
        """
        # --- 【Python 语法】检查关键词 ---
        # 统计回答中包含的预期关键词比例
        found_keywords = [kw for kw in expected_keywords if kw in actual_answer]
        recall = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

        result = {
            'query': query,
            'expected_keywords': expected_keywords,
            'found_keywords': found_keywords,
            'recall': recall,
            'passed': recall >= 0.5,  # 召回率 >= 50% 算通过
        }
        self.evaluation_results.append(result)
        return result

    def summary(self):
        """打印评估摘要"""
        if not self.evaluation_results:
            print("  没有评估结果")
            return

        passed = sum(1 for r in self.evaluation_results if r['passed'])
        total = len(self.evaluation_results)
        avg_recall = sum(r['recall'] for r in self.evaluation_results) / total

        print(f"\n  评估摘要:")
        print(f"    总测试数: {total}")
        print(f"    通过数: {passed}")
        print(f"    通过率: {passed/total*100:.1f}%")
        print(f"    平均召回率: {avg_recall*100:.1f}%")


# ============================================================
# 4. 快速演示
# ============================================================

def demo_production_rag():
    """
    生产级 RAG 系统演示

    整合缓存、混合搜索和评估
    """
    print("\n>>> 生产级 RAG 系统演示")

    # 创建测试文档
    documents = [
        Document(text="LlamaIndex 是一个用于构建 LLM 应用的框架。"
                      "它支持文档加载、文本分块、嵌入、索引和查询。"),
        Document(text="RAG（检索增强生成）结合了信息检索和文本生成。"
                      "先用向量数据库检索相关文档，再将文档作为上下文提供给 LLM。"),
        Document(text="向量数据库专门用于存储和检索高维向量。"
                      "Milvus、ChromaDB、FAISS 是常用的向量数据库。"),
    ]

    # 构建索引
    index = VectorStoreIndex.from_documents(documents)

    # --- 【Python 语法】创建缓存 ---
    cache = QueryCache(max_size=100, ttl_seconds=300)  # 5 分钟过期

    # 测试查询
    questions = [
        ("什么是 LlamaIndex?", ["LlamaIndex", "框架"]),
        ("RAG 的工作原理是什么?", ["检索", "生成", "LLM"]),
    ]

    for question, keywords in questions:
        # 先查缓存
        cached_answer = cache.get(question)
        if cached_answer:
            print(f"\n  [缓存命中] {question}")
            print(f"  回答: {cached_answer}")
            continue

        # 缓存未命中，调用查询引擎
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        answer = response.response

        # 存入缓存
        cache.put(question, answer)
        print(f"\n  [新查询] {question}")
        print(f"  回答: {answer}")

    # 评估
    evaluator = RagEvaluator()
    for question, keywords in questions:
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        evaluator.evaluate(question, keywords, response.response)

    evaluator.summary()


# ============================================================
# 主程序
# ============================================================

def main():
    """
    主函数
    """
    demo_production_rag()

    print("\n" + "=" * 60)
    print("  OK 生产级 RAG 系统完成！")
    print("=" * 60)
    print("""
恭喜！你已经完成了 LlamaIndex 的全部学习路径。

你现在掌握了：
✓ Python 基础语法
✓ LlamaIndex 核心概念
✓ 文档加载与分块
✓ Embedding 与向量存储
✓ 索引构建与检索
✓ 查询引擎
✓ 完整 RAG 管道
✓ 生产级 RAG 最佳实践

下一步建议：
1. 用你自己的文档替换示例文档
2. 尝试接入真实的 API（OpenAI / DashScope / 本地模型）
3. 部署为 Web 服务（FastAPI + Gradio）
4. 探索 LlamaIndex 的高级功能（工作流、工具调用等）
    """)


if __name__ == "__main__":
    main()
