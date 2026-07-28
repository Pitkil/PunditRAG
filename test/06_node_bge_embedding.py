import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.state import create_default_state


if __name__ == "__main__":
    """
    BGE-M3 向量化节点本地测试方法
    功能：模拟上游节点输出的 chunks 数据，独立测试 node_bge_embedding 节点
    适用场景：本地开发、调试、单节点功能验证
    测试前准备：
        1. 确保项目环境变量配置完成
        2. 确保 BGE-M3 模型路径、设备配置正确
        3. 确保当前 Python 环境可以正常加载 torch 和向量模型
    使用方法：
        直接执行该文件即可
    """
    from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding

    test_state = create_default_state(
        task_id="test_task_embedding_001",
        chunks=[
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf",
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf",
            },
        ],
    )

    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        result_state = node_bge_embedding(test_state)
        result_chunks = result_state.get("chunks", [])

        logger.info("=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")
        logger.info(f"返回的结果：{result_chunks}")

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===错误原因：{str(e)}", exc_info=True)
        logger.warning("排查提示：请检查 BGE-M3 模型路径、显存是否充足、环境变量配置是否正确")
