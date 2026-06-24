"""
数据库初始化脚本。
"""
import logging
import sys

from database import db_manager
from config import config

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def main():
    """主函数，初始化数据库。"""
    try:
        logger.info("开始初始化数据库...")
        db_manager.init_database()
        logger.info("数据库初始化成功！")
        
        # 测试数据库连接
        logger.info("测试数据库连接...")
        test_conversations = db_manager.get_conversation_history(limit=1)
        logger.info(f"数据库连接测试成功，当前有 {len(test_conversations)} 条历史记录")
        
        print("✅ 数据库初始化完成！")
        print("📊 数据库表结构已创建")
        print("🔗 数据库连接测试通过")
        print("\n现在可以启动应用程序：")
        print("  python start_all.py  # 一键启动")
        print("  或")
        print("  python LocalQwen3.py       # 仅启动API")
        print("  python gradio_app.py # 仅启动Web界面")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        print(f"❌ 数据库初始化失败: {e}")
        print("\n请检查：")
        print("1. PostgreSQL服务是否运行")
        print("2. 数据库配置是否正确（.env文件）")
        print("3. 数据库用户权限是否足够")
        sys.exit(1)


if __name__ == "__main__":
    main()