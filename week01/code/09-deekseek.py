"""
文件名：deepseek_via_openai_sdk.py
描述：使用标准 OpenAI SDK 接入并调用 DeepSeek-V3 或 DeepSeek-R1 大模型
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. 尝试从本地加载 .env 配置文件（推荐）
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
deepseekApiKey = os.getenv('DEEPSEEK_API_KEY')

def call_deepseek_demo():
    print("=== 开始通过 OpenAI SDK 接入 DeepSeek ===")

    # 2. 核心对齐：初始化 OpenAI 客户端
    # 无论你在 .env 中配置，还是在这里硬编码，必须修改以下两个核心参数：
    client = OpenAI(
        # 核心 1：必须把网关地址（Base URL）指向 DeepSeek 官方服务器
        base_url="https://deepseek.com",

        # 核心 2：必须传入你从 DeepSeek 开放平台申请到的 sk- 开头的密钥
        # 优先读取环境变量 OPENAI_API_KEY，如果没有则使用右侧的硬编码备用
        api_key=deepseekApiKey
    )

    try:
        # 3. 发起流式网络请求（Chat Completion）
        # 注意：由于 DeepSeek 兼容 OpenAI 协议，这里的方法名、传参格式和 OpenAI 完全一模一样！
        response = client.chat.completions.create(
            # 💡 选型：模型名称可以换成 "deepseek-chat" (V3模型) 或 "deepseek-reasoner" (R1深度思考模型)
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个资深的 Java 后端技术专家，用专业、简洁的语言回答问题。"},
                {"role": "user", "content": "请用一句话解释，为什么可以用 OpenAI 的 SDK 调你的接口？"}
            ],
            stream=False  # 为了在控制台一次性看清结果，这里先设为 False。如果需要打字机效果可以设为 True
        )

        # 4. 解析并打印返回数据（结构和 OpenAI 100% 对齐）
        ai_reply = response.choices[0].message.content
        print("\n🤖 DeepSeek 成功返回：")
        print(ai_reply)

    except Exception as e:
        print(f"\n❌ 请求发生异常（请检查你的网络代理或 Key 是否有效）：\n{e}")


if __name__ == "__main__":
    # 执行测试
    call_deepseek_demo()
