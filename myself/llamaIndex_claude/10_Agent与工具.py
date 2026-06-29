"""
===============================================================================
 第 10 课：Agent + Tools / 智能体与工具
===============================================================================

【这一课学什么？】
  前面的课程都是"用户问 → AI 查知识库 → AI 答"。
  但真实场景中，AI 可能需要：
    - 查数据库（工具1）
    - 调用 API（工具2）
    - 做数学计算（工具3）
    - 发送邮件（工具4）

  Agent 就是让 LLM 学会"自主决定使用哪个工具"的智能体。

【类比 Java】
  Agent ≈ 策略模式 + 工厂模式
  - LLM 是"大脑"，决定调用哪个工具
  - Tool 是"手"，执行具体操作
  - Agent Loop 是"循环"，反复思考 → 行动 → 观察

  伪代码类比：
    // Agent 的工作循环
    while (!hasAnswer) {
        Thought thought = llm.think(currentContext);  // 思考
        if (thought.needTool) {
            Tool tool = toolFactory.get(thought.toolName);
            Result result = tool.execute(thought.args);
            context = context.with(result);           // 观察
        } else {
            return llm.answer(context);               // 回答
        }
    }

【核心概念】
  Agent 的组成：
    - AgentRunner → 智能体本体（思考+行动循环）
    - Tool → 工具（函数封装）
    - ToolRunner → 工具执行器

  常用 Tool：
    - FunctionTool → 把任意 Python 函数变成 Tool
    - QueryEngineTool → 把 QueryEngine 变成 Tool
    - RetrieverTool → 把 Retriever 变成 Tool

【运行方式】
  在 week03 的虚拟环境中执行：
    cd week03
    source .venv/bin/activate
    python ../myself/llamaIndex_claude/10_Agent与工具.py

【前置知识】
  第 1-9 课的所有知识
"""

import os
from llama_index.core import Settings


# ============================================================================
# 第 1 节：什么是 Agent？
# ============================================================================

def demo_what_is_agent():
    """
    Agent = LLM + Tools + Planning Loop

    传统 RAG：
      用户问 → 检索知识库 → LLM 回答

    Agent RAG：
      用户问 → LLM 思考 → 选择工具 → 执行 → 观察结果 → 再思考 → ... → 回答

    类比 Java 的设计模式：
      传统 RAG = 命令模式（固定的执行流程）
      Agent RAG = 责任链模式（多个组件协作，动态决定顺序）
    """
    print("=" * 60)
    print("【什么是 Agent？】")
    print("=" * 60)

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                    Agent 工作原理                            │
  │                                                             │
  │   用户问题                                                   │
  │      │                                                       │
  │      ▼                                                       │
  │   ┌─────────┐                                               │
  │   │  LLM    │ ←── 思考：我需要用什么工具？                    │
  │   └────┬────┘                                               │
  │        │                                                     │
  │   ┌────┴────┐                                               │
  │   │ 有工具？│                                               │
  │   └────┬────┘                                               │
  │   yes  │  no                                               │
  │   ▼    │  ▼                                                 │
  │ 执行   │  直接回答                                          │
  │ 工具   │                                                    │
  │   │    │                                                    │
  │   ▼    │                                                    │
  │ 观察   │                                                    │
  │ 结果   │                                                    │
  │   │    │                                                    │
  │   └────┘                                                    │
  │        │                                                    │
  │        ▼                                                    │
  │   还需要工具？ ──yes──→ 回到 LLM 思考                        │
  │        │                                                    │
  │       no                                                   │
  │        │                                                    │
  │        ▼                                                    │
  │   生成最终回答                                               │
  └─────────────────────────────────────────────────────────────┘

  经典案例：
    用户: "帮我查一下张三的工资，然后发邮件给他"
    Agent 的思考过程:
      1. "我需要查工资 → 用 DatabaseTool"
      2. "查到了，张三工资20000 → 现在发邮件 → 用 EmailTool"
      3. "邮件发了 → 回答用户"
    """)


# ============================================================================
# 第 2 节：FunctionTool — 把函数变成工具
# ============================================================================

def demo_function_tool():
    """
    FunctionTool 是最基础的 Tool 类型

    它可以把任意 Python 函数包装成 LLM 可调用的工具。
    类比 Java：
      // 普通函数
      public int add(int a, int b) { return a + b; }

      // 包装成 Tool
      Tool addTool = FunctionTool.fromDefaults(add);
      // LLM 可以调用: addTool.execute(a=3, b=5) → 8
    """
    print("=" * 60)
    print("【FunctionTool — 把函数变成工具】")
    print("=" * 60)

    from llama_index.core.tools import FunctionTool

    # 第 1 步：定义一个普通函数
    print("\n  --- 第 1 步：定义工具函数 ---")

    def calculate_salary(base_salary, bonus, tax_rate=0.1):
        """
        计算税后工资

        参数：
          base_salary → 基本工资
          bonus       → 奖金
          tax_rate    → 税率（默认 10%）

        返回：
          税后总收入

        类比 Java:
          public double calculateSalary(double base, double bonus, double tax) {
              return (base + bonus) * (1 - tax);
          }
        """
        total = (base_salary + bonus) * (1 - tax_rate)
        return round(total, 2)

    def get_employee_info(name, department):
        """
        查询员工信息（模拟数据库查询）

        类比 Java:
          public Employee getEmployee(String name, String dept) {
              return employeeRepo.findByNameAndDept(name, dept);
          }
        """
        # 模拟数据库
        employees = {
            ("张三", "技术部"): {"salary": 25000, "bonus": 5000, "level": "P7"},
            ("李四", "市场部"): {"salary": 18000, "bonus": 3000, "level": "P6"},
            ("王五", "人事部"): {"salary": 16000, "bonus": 2000, "level": "P5"},
        }
        return employees.get((name, department), "未找到该员工")

    # 第 2 步：把函数包装成 Tool
    print("\n  --- 第 2 步：包装成 Tool ---")

    # from_defaults() 的参数：
    #   fn          → 要包装的函数
    #   name        → 工具名称（LLM 用它来选择）
    #   description → 工具描述（LLM 用它来决定何时使用）
    calc_tool = FunctionTool.from_defaults(
        fn=calculate_salary,
        name="calculate_salary",
        description="计算税后工资，需要传入基本工资、奖金和税率"
    )

    info_tool = FunctionTool.from_defaults(
        fn=get_employee_info,
        name="get_employee_info",
        description="查询员工信息，需要传入姓名和部门"
    )

    print(f"    ✓ 创建了工具: {calc_tool.metadata.name}")
    print(f"    ✓ 创建了工具: {info_tool.metadata.name}")

    # 第 3 步：测试工具
    print("\n  --- 第 3 步：测试工具 ---")

    # 直接调用工具
    result1 = calc_tool.run(base_salary=20000, bonus=5000, tax_rate=0.1)
    print(f"    calculate_salary(20000, 5000, 0.1) = {result1.output}")

    result2 = info_tool.run(name="张三", department="技术部")
    print(f"    get_employee_info('张三', '技术部') = {result2.output}")


# ============================================================================
# 第 3 节：QueryEngineTool — 把查询引擎变成工具
# ============================================================================

def demo_query_engine_tool():
    """
    QueryEngineTool 让 LLM 可以调用 QueryEngine

    这意味着 Agent 可以访问知识库！

    类比 Java：
      // 把 Service 方法包装成 Tool
      Tool knowledgeTool = FunctionTool.fromDefaults(queryEngine::query);
      // LLM 可以调用: knowledgeTool.execute("年假几天？")
    """
    print("=" * 60)
    print("【QueryEngineTool — 把查询引擎变成工具】")
    print("=" * 60)

    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.tools import QueryEngineTool

    # 准备知识库文档
    documents = [
        Document(text="员工每年享有5天带薪年假，满10年10天，满20年15天。"),
        Document(text="病假需提供二级以上医院诊断证明，按基本工资80%发放。"),
        Document(text="婚假15天，产假98天，陪产假15天。"),
    ]

    # 创建 Index 和 QueryEngine
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 把 QueryEngine 包装成 Tool
    # 参数说明：
    #   query_engine  → 要包装的查询引擎
    #   name          → 工具名称
    #   description   → 工具描述
    knowledge_tool = QueryEngineTool(
        query_engine=query_engine,
        name="knowledge_base",
        description="查询公司制度知识库，包括年假、病假、婚假等信息"
    )

    print(f"    ✓ 创建了工具: {knowledge_tool.metadata.name}")
    print(f"    描述: {knowledge_tool.metadata.description}")

    # 测试工具
    print("\n  --- 测试工具 ---")
    result = knowledge_tool.run("年假几天？")
    print(f"    问: 年假几天？")
    print(f"    答: {result.output[:80]}...")


# ============================================================================
# 第 4 节：ReAct Agent — 最经典的 Agent 模式
# ============================================================================

def demo_react_agent():
    """
    ReAct Agent = Reasoning + Acting

    它的核心思想：让 LLM 在思考和行动之间反复循环，
    直到找到足够信息来回答问题。

    类比 Java 的状态机：
      State: THINKING → ACTING → OBSERVING → THINKING → ... → ANSWERING
    """
    print("=" * 60)
    print("【ReAct Agent — 思考+行动的循环】")
    print("=" * 60)

    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool

    # 定义工具
    def add_numbers(a: int, b: int) -> int:
        """计算两个数的和"""
        return a + b

    def multiply_numbers(a: int, b: int) -> int:
        """计算两个数的乘积"""
        return a * b

    add_tool = FunctionTool.from_defaults(add_numbers)
    mul_tool = FunctionTool.from_defaults(multiply_numbers)

    print("\n  --- 创建 Agent ---")
    print("    工具: add_numbers, multiply_numbers")

    # 创建 ReAct Agent
    # 参数说明：
    #   tools       → 可用的工具列表
    #   llm         → 使用的 LLM
    #   max_iterations → 最大思考/行动次数（防止死循环）
    agent = ReActAgent.from_tools(
        tools=[add_tool, mul_tool],
        max_iterations=10,
    )

    print("    ✓ Agent 创建完成")

    # 测试：让 Agent 解决一个需要多步计算的问题
    print("\n  --- 测试 Agent ---")
    print("    问: (3 + 5) * 2 = ?")
    print("    Agent 的思考过程:")
    print("      1. 看到乘法 → 需要先算加法")
    print("      2. 调用 add_numbers(3, 5) → 8")
    print("      3. 调用 multiply_numbers(8, 2) → 16")
    print("      4. 回答: 16")

    chat_engine = agent.chat_stream("计算 (3 + 5) * 2 的结果")
    for delta in chat_engine.response_gen:
        print(f"    {delta}", end="", flush=True)
    print()


# ============================================================================
# 第 5 节：Agent 的完整工作流
# ============================================================================

def demo_agent_workflow():
    """
    演示 Agent 的完整工作流程

    这是从工具定义到 Agent 响应的端到端示例。
    """
    print("=" * 60)
    print("【Agent 完整工作流】")
    print("=" * 60)

    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool, QueryEngineTool
    from llama_index.core import Document
    from llama_index.core.indices.vector_store import VectorStoreIndex

    # 第 1 步：定义工具
    print("\n  --- 第 1 步：定义工具 ---")

    def get_weather(city: str) -> str:
        """查询城市天气"""
        weather_data = {
            "北京": "晴天，15°C",
            "上海": "小雨，12°C",
            "深圳": "多云，25°C",
        }
        return weather_data.get(city, "未知城市")

    def convert_temperature(celsius: float) -> float:
        """摄氏度转华氏度"""
        return round(celsius * 9 / 5 + 32, 1)

    weather_tool = FunctionTool.from_defaults(
        fn=get_weather,
        name="get_weather",
        description="查询指定城市的天气情况"
    )

    temp_tool = FunctionTool.from_defaults(
        fn=convert_temperature,
        name="convert_temperature",
        description="将摄氏度转换为华氏度"
    )

    print(f"    ✓ 创建了 2 个工具: get_weather, convert_temperature")

    # 第 2 步：创建 Agent
    print("\n  --- 第 2 步：创建 Agent ---")
    agent = ReActAgent.from_tools(
        tools=[weather_tool, temp_tool],
        max_iterations=5,
    )
    print("    ✓ Agent 创建完成")

    # 第 3 步：测试
    print("\n  --- 第 3 步：测试 Agent ---")

    test_queries = [
        "北京今天天气怎么样？",
        "把 25 摄氏度转成华氏度",
        "北京和上海哪个更热？",
    ]

    for query in test_queries:
        print(f"\n    用户: {query}")
        response = agent.chat(query)
        print(f"    Agent: {response.response}")


# ============================================================================
# 第 6 节：Agent vs 普通 QueryEngine
# ============================================================================

def demo_agent_vs_query_engine():
    """
    对比 Agent 和普通 QueryEngine 的能力差异
    """
    print("=" * 60)
    print("【Agent vs QueryEngine 对比】")
    print("=" * 60)

    print("""
  ┌────────────────────┬─────────────────────┬────────────────────┐
  │ 特性               │ QueryEngine         │ Agent              │
  ├────────────────────┼─────────────────────┼────────────────────┤
  │ 输入               │ 用户问题             │ 用户问题           │
  │ 处理方式           │ 检索 → 合成回答      │ 思考 → 选工具 → 执行 → 思考 → ... │
  │ 能做什么           │ 回答知识库问题       │ 回答 + 计算 + 查天气 + 调用API │
  │ 可控性             │ 确定性强            │ 动态决策，可能有不确定性 │
  │ 速度               │ 快                  │ 慢（多次 LLM 调用） │
  │ 成本               │ 低                  │ 高（多次 LLM 调用） │
  │ Java 类比          │ Service 方法        │ 策略模式 + 工厂    │
  └────────────────────┴─────────────────────┴────────────────────┘

  选择建议：
    - 只需要回答知识库问题 → QueryEngine（简单高效）
    - 需要调用外部工具/API → Agent（灵活强大）
    - 两者结合 → 用 QueryEngineTool 把 QueryEngine 包装成 Agent 的工具
    """)


# ============================================================================
# 第 7 节：本课总结
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│                          第 10 课总结                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【核心知识点】                                                           │
│  1. Agent = LLM + Tools + Planning Loop                                 │
│  2. FunctionTool 把任意函数变成 LLM 可调用的工具                          │
│  3. QueryEngineTool 把查询引擎变成工具                                    │
│  4. ReAct Agent 是最经典的 Agent 模式（思考→行动→观察→思考...）           │
│  5. Agent 可以调用多个工具解决复杂问题                                    │
│  6. Agent 比 QueryEngine 更强，但更慢更贵                                │
│                                                                         │
│  【关键代码模板】                                                         │
│                                                                         │
│  # 定义工具                                                              │
│  from llama_index.core.tools import FunctionTool                        │
│                                                                       │
│  def my_function(param1, param2):                                       │
│      # 工具逻辑                                                        │
│      return result                                                    │
│                                                                       │
│  tool = FunctionTool.from_defaults(                                      │
│      fn=my_function,                                                   │
│      name="my_tool",                                                   │
│      description="工具的描述"                                           │
│  )                                                                      │
│                                                                       │
│  # 创建 Agent                                                           │
│  from llama_index.core.agent import ReActAgent                          │
│                                                                       │
│  agent = ReActAgent.from_tools(                                         │
│      tools=[tool1, tool2],                                             │
│      max_iterations=10                                                 │
│  )                                                                      │
│                                                                       │
│  # 与 Agent 对话                                                        │
│  response = agent.chat("用户的问题")                                     │
│  print(response.response)                                               │
│                                                                       │
│  # 把 QueryEngine 变成 Agent 的工具                                     │
│  from llama_index.core.tools import QueryEngineTool                     │
│                                                                       │
│  knowledge_tool = QueryEngineTool(                                      │
│      query_engine=index.as_query_engine(),                              │
│      name="knowledge_base",                                             │
│      description="查询知识库"                                           │
│  )                                                                      │
│                                                                       │
│  agent = ReActAgent.from_tools(tools=[knowledge_tool, ...])             │
│                                                                         │
│  【下一课预告】                                                           │
│  第 11 课：综合项目 — 完整 RAG 系统                                      │
│  类比 Java：Spring Boot 全栈应用                                        │
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
║   第 10 课：Agent + Tools / 智能体与工具                  ║
║                                                          ║
║   面向 Java 程序员                                       ║
║                                                          ║
║   本节内容：                                              ║
║   1. 什么是 Agent？                                      ║
║   2. FunctionTool — 把函数变成工具                       ║
║   3. QueryEngineTool — 把查询引擎变成工具                ║
║   4. ReAct Agent — 思考+行动的循环                       ║
║   5. Agent 完整工作流                                    ║
║   6. Agent vs QueryEngine 对比                           ║
║   7. 总结                                               ║
║                                                          ║
║   前置知识：第 1-9 课                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

    # 依次运行各个演示
    print("━━━ 第 1 节：什么是 Agent？ ━━━")
    demo_what_is_agent()

    print("\n━━━ 第 2 节：FunctionTool ━━━")
    demo_function_tool()

    print("\n━━━ 第 3 节：QueryEngineTool ━━━")
    demo_query_engine_tool()

    print("\n━━━ 第 4 节：ReAct Agent ━━━")
    demo_react_agent()

    print("\n━━━ 第 5 节：Agent 完整工作流 ━━━")
    demo_agent_workflow()

    print("\n━━━ 第 6 节：Agent vs QueryEngine ━━━")
    demo_agent_vs_query_engine()

    print("\n🎉 第 10 课完成！")
    print("   建议下一步：阅读 week03/code/p32-agent-tools.ipynb")
