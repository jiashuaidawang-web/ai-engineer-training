"""
===========================================================================
 GraphRAG Demo — 公司控股关系图谱演示
===========================================================================
【给 Java 程序员的 Python 速查】
 1. dataclass  ≈ @Data + class（自动实现 __init__/__repr__/__eq__）
 2. async/await  ≈ CompletableFuture.thenApplyAsync(...)，方法前必须加 async
 3. with self.driver.session() as session:  ≈ try-with-resources，退出自动 close session
 4. List[str]  ≈ List<String>；Dict  ≈ Map<String, Object>
 5. f"hello {name}"  ≈ String.format("hello %s", name)
 6. result.get("key", [])  ≈ map.getOrDefault("key", Collections.emptyList())
 7. self.xxx  ≈ this.xxx（Python 没有 implicit this，必须显式写 self）
===========================================================================
"""

import asyncio          # 异步事件循环（类似 Java 的 ExecutorService / CompletableFuture）
import json              # JSON 序列化/反序列化
import logging
from dataclasses import dataclass   # 注解：告诉 Python 这是一个"纯数据容器类"
from typing import List, Dict, Tuple  # 类型提示（Java 里是强类型，Python 是可选的"注释"）

from neo4j import GraphDatabase     # Neo4j Python 驱动（类似 Java 的 neo4j-java-driver）
from neo4j_graphrag.llm import OpenAILLM  # Neo4j 官方提供的 LLM 封装（内部调用 OpenAI 兼容 API）

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity / Relationship：相当于 Java 里的 @Value class 或 Lombok @Data class
# 只有字段，没有行为。Python 的 dataclass 会自动生成 __init__, __repr__, __eq__ 等
# ---------------------------------------------------------------------------
@dataclass
class Entity:
    """实体节点 — 对应图数据库中的 Node"""
    name: str       # 实体名称，如 "A公司"
    type: str       # 实体类型，如 "Company"


@dataclass
class Relationship:
    """关系边 — 对应图数据库中的 Relationship"""
    source: str     # 起始节点名称
    target: str     # 目标节点名称
    type: str       # 关系类型，如 "CONTROLS"（控股）


# ===========================================================================
# GraphRAG：整个演示的核心类
# 职责：把自然语言文本 → LLM 抽取 → Neo4j 图数据库 → 多跳查询问答
# 类比 Java：一个 Service 类，依赖注入 Driver + LLM
# ===========================================================================
class GraphRAG:
    """GraphRAG 核心类 - 公司控股关系演示

    【架构类比】
    ┌─────────────┐     ┌──────────┐     ┌──────────┐
    │  自然语言文本 │ ──▶ │  LLM抽取  │ ──▶ │ Neo4j图谱 │
    └─────────────┘     └──────────┘     └──────────┘
                              │                  │
                              ▼                  ▼
                         JSON 结构           Cypher 查询
                         实体/关系           多跳推理/问答
    """

    # -------------------------------------------------------------------
    # __init__：构造函数
    # driver      ：Neo4j 数据库连接池（类似 Java 的 DataSource）
    # llm_json    ：负责"结构化输出"的 LLM（要求返回 JSON）
    # llm_text    ：负责"自由文本生成"的 LLM
    # -------------------------------------------------------------------
    def __init__(self, driver, llm_json, llm_text):
        self.driver = driver            # self ≈ this
        self.llm_json = llm_json        # 用于 extract 阶段的 JSON 模式 LLM
        self.llm_text = llm_text        # 用于 query 阶段的对话式 LLM

    # -------------------------------------------------------------------
    # extract_entities：步骤1 — 让 LLM 从文本中识别出公司实体
    # 类比 Java：public List<Entity> extractEntities(String text)
    # -------------------------------------------------------------------
    async def extract_entities(self, text: str) -> List[Entity]:
        """步骤1: 提取公司实体

        Args:
            text: 包含公司信息的一段自然语言文本

        Returns:
            提取到的 Entity 对象列表

        流程：构造 Prompt → 调 LLM → 解析 JSON → 转成 Entity 对象
        """
        # 构造 Prompt：告诉 LLM 要做什么、返回什么格式
        prompt = f"""
        从文本中提取公司实体：

        文本：{text}

        返回JSON格式：
        {{
            "entities": [
                {{"name": "公司名", "type": "Company"}}
            ]
        }}
        """

        # 调用 LLM（async 方法，类似 Java 的 return future.join()）
        response = await self.llm_json.ainvoke(prompt)
        # LLM 返回的内容在 response.content 里，是 JSON 字符串
        result = json.loads(response.content)

        # 列表推导式：[表达式 for 变量 in 可迭代对象]
        # 类比 Java: result.get("entities", []).stream()
        #                          .map(e -> new Entity(e.name, e.type))
        #                          .toList()
        entities = [Entity(e["name"], e["type"]) for e in result.get("entities", [])]
        print(f" 提取到 {len(entities)} 个公司实体")
        return entities

    # -------------------------------------------------------------------
    # extract_relationships：步骤2 — 让 LLM 从文本中识别出控股关系
    # -------------------------------------------------------------------
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """步骤2: 提取控股关系

        Args:
            text: 原始自然语言文本
            entities: 上一步提取出的公司实体列表

        Returns:
            Relationship 对象列表，每条表示 "A CONTROLS B"
        """
        # 拿到所有公司名的列表，用于后续过滤
        entity_names = [e.name for e in entities]

        prompt = f"""
        从文本中提取公司间的控股关系：

        文本：{text}
        公司：{entity_names}

        返回JSON格式：
        {{
            "relationships": [
                {{"source": "母公司", "target": "子公司", "type": "CONTROLS"}}
            ]
        }}
        """

        response = await self.llm_json.ainvoke(prompt)
        result = json.loads(response.content)

        # 过滤：只保留 source 和 target 都存在于实体列表中的关系
        # （防止 LLM 幻觉，抽出了不存在的公司名）
        relationships = []
        for r in result.get("relationships", []):
            if r["source"] in entity_names and r["target"] in entity_names:
                relationships.append(Relationship(r["source"], r["target"], r["type"]))

        print(f" 提取到 {len(relationships)} 个控股关系")
        return relationships

    # -------------------------------------------------------------------
    # build_graph：步骤3 — 把提取结果写入 Neo4j 图数据库
    # -------------------------------------------------------------------
    async def build_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """步骤3: 构建公司控股图谱

        将内存中的 Entity 和 Relationship 持久化到 Neo4j 数据库。

        Args:
            entities: 公司实体列表
            relationships: 控股关系列表

        注意：with self.driver.session() as session:
              这是 Python 的"上下文管理器"，等价于 Java 的 try-with-resources：
              try (Session session = driver.session()) { ... }
              离开 with 块后自动关闭 session，释放资源。
        """
        with self.driver.session() as session:
            # 清空现有数据（每次重新演示，避免重复写入）
            session.run("MATCH (n) DETACH DELETE n")

            # 写入公司实体节点
            # MERGE = UPSERT（存在就匹配，不存在就创建）
            # 类比 Java: session.merge("MERGE (n:Company {name: $name})", params)
            for entity in entities:
                query = f"MERGE (n:{entity.type} {{name: $name}})"
                session.run(query, name=entity.name)

            # 写入控股关系边
            for rel in relationships:
                query = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[:{rel.type}]->(b)
                """
                session.run(query, source=rel.source, target=rel.target)

        print(f" 公司控股图谱构建完成")

    # -------------------------------------------------------------------
    # find_subsidiaries_with_path：图遍历 — 查找所有子公司及控股路径
    # -------------------------------------------------------------------
    def find_subsidiaries_with_path(self, parent_company: str) -> List[Dict]:
        """使用图遍历算法查找所有子公司及路径

        这是整个系统的核心查询 — 利用 Neo4j 的"路径模式匹配"做多跳推理。

        Args:
            parent_company: 母公司名称，如 "A公司"

        Returns:
            子公司列表，每个元素是 {'subsidiary': '...', 'depth': 1, 'path': ['A', 'B']}

        【Cypher 语句解读】（Neo4j 的查询语言，类似 SQL）
        ┌──────────────────────────────────────────────────────────────────┐
        │ MATCH path =                                                   │
        │   (parent:Company {name: $parent_name})                        │
        │   -[:CONTROLS*1..]->                                           │   ← ★ 关键：*1.. 表示"1跳到无穷跳" │
        │   (subsidiary:Company)                                         │
        │                                                                │
        │ 这条语句的意思是：                                               │
        │ "从 A公司 出发，沿着 CONTROLS 关系走 1 步、2 步、3 步...       │
        │  把所有能到达的节点都找出来"                                     │
        │                                                                │
        │ 类比 Java/SQL:                                                 │
        │   WITH RECURSIVE cte AS (                                      │
        │     SELECT name, 1 as depth FROM Company WHERE name = 'A'      │
        │     UNION ALL                                                  │
        │     SELECT c.name, ct.depth+1                                  │
        │     FROM Control ct JOIN Company c ON ct.child_id=c.id          │
        │     WHERE ct.parent_id IN (SELECT id FROM cte)                 │
        │   )                                                            │
        │   SELECT * FROM cte;                                           │
        └──────────────────────────────────────────────────────────────────┘
        """
        with self.driver.session() as session:
            # 使用 Cypher 的路径查询功能实现多跳推理
            query = """
            MATCH path = (parent:Company {name: $parent_name})-[:CONTROLS*1..]->(subsidiary:Company)
            RETURN subsidiary.name as subsidiary,
                   length(path) as depth,
                   [node in nodes(path) | node.name] as path_nodes
            ORDER BY depth, subsidiary.name
            """

            # 执行查询，返回一个 Result 对象（可迭代，类似 Java 的 ResultSet）
            result = session.run(query, parent_name=parent_company)
            subsidiaries = []

            # 遍历结果集（类似 for (Record record : result)）
            for record in result:
                subsidiaries.append({
                    'subsidiary': record['subsidiary'],
                    'depth': record['depth'],
                    'path': record['path_nodes']
                })

            return subsidiaries

    # -------------------------------------------------------------------
    # visualize_control_structure：控制台可视化控股树
    # -------------------------------------------------------------------
    def visualize_control_structure(self, parent_company: str):
        """可视化控股结构（在控制台打印）

        按层级分组展示控股树，类似：
            第1层子公司:
              • B公司
                路径: A公司 → B公司
              • D公司
                路径: A公司 → D公司
            第2层子公司:
              • C公司
                路径: A公司 → B公司 → C公司
        """
        subsidiaries = self.find_subsidiaries_with_path(parent_company)

        print(f" {parent_company} 的控股结构:")
        print("=" * 50)

        if not subsidiaries:
            print(f"   {parent_company} 没有子公司")
            return

        # 按层级分组：{1: [B公司, D公司], 2: [C公司, E公司], ...}
        # 类比 Java: Map<Integer, List<Subsidiary>> levels = new TreeMap<>();
        levels = {}
        for sub in subsidiaries:
            depth = sub['depth']
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(sub)

        # 按深度从小到大遍历并打印
        for depth in sorted(levels.keys()):
            print(f"第{depth}层子公司:")
            for sub in levels[depth]:
                path_str = " → ".join(sub['path'])
                print(f"   • {sub['subsidiary']}")
                print(f"     路径: {path_str}")

    # -------------------------------------------------------------------
    # query_graph：智能问答 — 根据用户问题返回答案
    # -------------------------------------------------------------------
    async def query_graph(self, question: str) -> str:
        """步骤4: 智能问答 - 支持多跳推理

        根据问题的关键词，选择不同的处理策略：

        策略1: 问题含"子公司" → 精确图查询（走 find_subsidiaries_with_path）
        策略2: 问题含"层级"/"多少层" → 统计查询
        策略3: 其他 → 通用查询（拉取图谱片段，丢给 LLM 自由回答）

        Args:
            question: 用户的问题，如 "A公司的子公司有哪些？"

        Returns:
            答案字符串

        类比 Java：
        public String queryGraph(String question) {
            if (question.contains("子公司")) { ... }
            else if (question.contains("层级")) { ... }
            else { ... }
        }
        """
        # ---- 策略1：提取"X公司的子公司"中的公司名 ----
        company_name = None
        if "子公司" in question:
            import re
            # 正则匹配：([A-Z]公司) 捕获一个大写字母+"公司"的组合
            # 例如 "A公司的子公司" → match.group(1) = "A公司"
            match = re.search(r'([A-Z]公司)的子公司', question)
            if match:
                company_name = match.group(1)

        if company_name:
            subsidiaries = self.find_subsidiaries_with_path(company_name)

            if subsidiaries:
                # 拼接答案
                answer_parts = [f"{company_name}的子公司包括:"]
                for sub in subsidiaries:
                    path_str = " → ".join(sub['path'])
                    answer_parts.append(f"• {sub['subsidiary']} (路径: {path_str})")
                return "\n".join(answer_parts)
            else:
                return f"{company_name}没有子公司"

        # ---- 策略2：处理层级问题 ----
        if "层级" in question or "多少层" in question:
            import re
            match = re.search(r'([A-Z]公司)', question)
            if match:
                company_name = match.group(1)
                subsidiaries = self.find_subsidiaries_with_path(company_name)
                if subsidiaries:
                    # 找出最大深度
                    max_depth = max(sub['depth'] for sub in subsidiaries)
                    return f"{company_name}共有{max_depth}层子公司，总计{len(subsidiaries)}个子公司"
                else:
                    return f"{company_name}没有子公司"

        # ---- 策略3：通用查询 — 把图谱片段喂给 LLM 让它自己组织语言 ----
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)-[r]-(m)
                RETURN n.name, type(r), m.name
                LIMIT 20
            """)

            context = []
            for record in result:
                # record 是元组，record[0]=节点A, record[1]=关系类型, record[2]=节点B
                context.append(f"{record[0]} {record[1]} {record[2]}")

        # 构造 Prompt，让 LLM 基于图谱信息回答问题
        prompt = f"""
        基于公司控股图谱回答问题：

        问题：{question}

        图谱信息：
        {chr(10).join(context)}    ← chr(10) 就是换行符 '\n'

        请简洁回答：
        """

        response = await self.llm_text.ainvoke(prompt)
        return response.content.strip()


# ===========================================================================
# demo()：演示函数（相当于 Java 的 public static void main）
# Python 脚本没有 class 包裹时，顶层函数就是入口点
# ===========================================================================
async def demo():
    """公司控股关系 GraphRAG 演示

    完整流程（类比 Java 的 main 方法）：
    1. 连接 Neo4j 数据库
    2. 初始化两个 LLM 客户端（一个用于 JSON 结构化抽取，一个用于对话生成）
    3. 创建 GraphRAG 服务实例
    4. 输入一段自然语言文本
    5. 抽取实体 → 抽取关系 → 构建图谱 → 可视化 → 问答
    """
    print(" 公司控股关系 GraphRAG 演示")
    print("=" * 50)

    # ---------- 1. 连接 Neo4j 数据库 ----------
    # GraphDatabase.driver() 创建连接（类似 JDBC DriverManager.getConnection()）
    # auth 参数是 (用户名, 密码)，类似 JDBC 的 username/password
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))

    # ---------- 2. 初始化 LLM ----------
    # llm_json：要求 LLM 返回 JSON（用于实体/关系抽取）
    llm_json = OpenAILLM(
        model_name="qwen-plus",                        # 使用阿里云通义千问-plus 模型
        model_params={                                 # 传给模型的参数
            "response_format": {"type": "json_object"},  # 强制返回 JSON
            "temperature": 0                             # temperature=0 让输出更确定
        }
    )

    # llm_text：自由文本生成（用于问答）
    llm_text = OpenAILLM(
        model_name="qwen-plus",
        model_params={"temperature": 0}
    )

    # ---------- 3. 创建 GraphRAG 服务实例 ----------
    # 把 driver + llm 注入到服务中（类似 Spring 的 @Autowired）
    graph_rag = GraphRAG(driver, llm_json, llm_text)

    # ---------- 4. 准备演示文本 ----------
    text = """
    A公司是一家大型集团公司。
    A公司控股B公司，持股比例为60%。
    A公司还控股D公司，持股比例为55%。
    B公司控股C公司，持股比例为70%。
    B公司控股E公司，持股比例为80%。
    C公司控股F公司，持股比例为65%。
    D公司控股G公司，持股比例为75%。
    """

    print(" 输入的公司控股信息:")
    print(text.strip())
    print()

    # ---------- 5. 执行完整流程 ----------
    try:
        # 步骤1: 让 LLM 从文本中识别出所有公司名
        print(" 步骤1: 提取公司实体")
        entities = await graph_rag.extract_entities(text)
        for e in entities:
            print(f"   • {e.name} ({e.type})")
        print()

        # 步骤2: 让 LLM 从文本中识别出控股关系
        print(" 步骤2: 提取控股关系")
        relationships = await graph_rag.extract_relationships(text, entities)
        for r in relationships:
            print(f"   • {r.source} --{r.type}--> {r.target}")
        print()

        # 步骤3: 把抽取结果写入 Neo4j
        print(" 步骤3: 构建公司控股图谱")
        await graph_rag.build_graph(entities, relationships)
        print()

        # 步骤4: 在控制台打印控股树
        print(" 步骤4: 可视化控股结构")
        graph_rag.visualize_control_structure("A公司")
        print()

        # 步骤5: 智能问答
        print(" 步骤5: 多跳推理智能问答")
        questions = [
            "A公司的子公司有哪些？",
            "B公司的子公司有哪些？",
            "A公司有多少层级的子公司？"
        ]

        for question in questions:
            print(f"    问: {question}")
            answer = await graph_rag.query_graph(question)
            print(f"    答: {answer}")
            print()

        # 额外演示：直接调用图遍历算法
        print(" 图遍历算法演示:")
        print("查找A公司的所有子公司及控股路径...")
        subsidiaries = graph_rag.find_subsidiaries_with_path("A公司")

        print(f" 多跳推理结果:")
        print(f"A公司共有 {len(subsidiaries)} 个子公司:")
        for sub in subsidiaries:
            path_str = " → ".join(sub['path'])
            print(f"   第{sub['depth']}层: {sub['subsidiary']}")
            print(f"   控股路径: {path_str}")
            print()

    except Exception as e:
        # 捕获所有异常并打印堆栈
        print(f" 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源（类似 Java 的 try-with-resources 或 finally 里 close）
        driver.close()
        print(" 演示完成")


# ===========================================================================
# 入口点
# ===========================================================================
if __name__ == "__main__":
    # Python 的 asyncio 事件循环
    # 因为 demo() 里有 await，所以不能直接调用，需要用 asyncio.run() 启动
    # 类比 Java: CompletableFuture.runAsync(() -> demo()).join();
    import nest_asyncio
    nest_asyncio.apply()       # 如果在 Jupyter 里运行，需要先 patch 事件循环
    asyncio.run(demo())         # 启动异步运行，进入 demo()
