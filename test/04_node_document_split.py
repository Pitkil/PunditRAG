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
    单元测试：联合 node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置 .env（MinIO/大模型环境） 2.存在测试 MD 文件 3.能导入节点
    测试流程：先运行图片处理 -> 再运行文档切分，验证端到端流程
    """
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img
    from app.import_process.agent.nodes.node_document_split import node_document_split

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试 MD 文件放入项目根目录的 output 目录下")
    else:
        test_state = create_default_state(
            task_id="test_task_123456",
            md_path=test_md_path,
            md_content="",
            file_title="hak180产品安全手册",
            local_dir=os.path.join(PROJECT_ROOT, "output"),
        )
        logger.info("开始本地测试 - MD 图片处理全流程")
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")

        logger.info("\n=== 开始执行文档切分节点集成测试 ===")
        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"测试成功：最终生成 {len(final_chunks)} 个有效 Chunk：{final_chunks}")
