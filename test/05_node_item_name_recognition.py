import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.state import create_default_state


if __name__ == "__main__":
    """
    关键词识别节点本地测试方法，此处识别商品名字
    功能：模拟 LangGraph 流程输入，独立测试 node_item_name_recognition 节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个 LangGraph 流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION 等）
        2. 确保大模型、Milvus、BGE-M3 服务均可正常访问
        3. 确保 prompt 模板已存在
    使用方法：
        直接执行该文件即可
    """
    from app.clients.milvus_utils import get_milvus_client
    from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition

    logger.info("=== 开始执行商品名称识别节点本地测试 ===")

    try:
        test_state = create_default_state(
            task_id="test_task_123456",
            file_title="华为Mate60 Pro手机使用说明书",
            chunks=[
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。",
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。",
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。",
                },
            ],
        )

        result_state = node_item_name_recognition(test_state)

        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            item_name = result_state.get("item_name")
            safe_name = item_name
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"],
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)
