import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.state import create_default_state


if __name__ == "__main__":
    """
    Milvus 导入节点本地测试方法
    功能：模拟上游向量化节点输出的 chunks 数据，独立测试 node_import_milvus 节点
    适用场景：本地开发、调试、单节点功能验证
    测试前准备：
        1. 确保项目环境变量配置完成
        2. 确保 Milvus 服务可正常访问
        3. 确保目标集合配置正确
    使用方法：
        直接执行该文件即可
    """
    from app.import_process.agent.nodes.node_import_milvus import node_import_milvus

    dim = 1024
    test_state = create_default_state(
        task_id="test_milvus_task",
        item_name="测试项目_Milvus",
        chunks=[
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            },
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            },
        ],
    )

    logger.info("=== 开始执行 Milvus 导入节点本地测试 ===")
    try:
        result_state = node_import_milvus(test_state)
        result_chunks = result_state.get("chunks", [])

        logger.info("=== Milvus 导入节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"当前 item_name：{result_state.get('item_name')}")
        logger.info(f"返回切片数量：{len(result_chunks)}")
        logger.info(f"返回结果：{result_chunks}")

    except Exception as e:
        logger.error(f"=== Milvus 导入节点本地测试失败 === 错误原因：{str(e)}", exc_info=True)
