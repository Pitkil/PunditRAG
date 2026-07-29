import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state


if __name__ == "__main__":
    """
    全流程本地测试方法
    功能：验证 PDF 导入 -> PDF 转 MD -> 文档切分 -> 商品名称识别 -> 向量化 -> Milvus 入库 的完整链路
    测试前准备：
        1. 确保项目环境变量配置完成
        2. 确保测试 PDF 文件存在
        3. 确保 MinerU、MinIO、大模型、BGE-M3、Milvus 服务可正常访问
    使用方法：
        直接执行该文件即可
    """
    from app.utils.path_util import PROJECT_ROOT as APP_PROJECT_ROOT

    logger.info("===== 开始执行导入全流程测试 =====")

    test_pdf_name = os.path.join("doc", "万用表RS-12的使用.pdf")
    test_pdf_path = os.path.join(APP_PROJECT_ROOT, test_pdf_name)
    test_output_dir = os.path.join(APP_PROJECT_ROOT, "output")
    os.makedirs(test_output_dir, exist_ok=True)

    if not os.path.exists(test_pdf_path):
        logger.error(f"全流程测试失败：测试 PDF 文件不存在，路径：{test_pdf_path}")
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的 doc 文件夹中")
    else:
        test_state = create_default_state(
            task_id="test_import_workflow_001",
            local_file_path=test_pdf_path,
            local_dir=test_output_dir,
            is_pdf_read_enabled=False,
            is_md_read_enabled=False,
        )

        try:
            logger.info(f"测试任务启动，PDF 文件路径：{test_pdf_path}")
            logger.info(f"中间文件输出目录：{test_output_dir}")
            logger.info("开始执行全流程节点，依次执行：entry -> pdf2md -> split -> item_name -> embedding -> milvus")

            final_state = None
            for step in kb_import_app.stream(test_state, stream_mode="updates"):
                current_node = next(iter(step)) if step else "未知节点"
                logger.info(f"节点执行完成：{current_node}")
                final_state = step.get(current_node, {})

            if final_state:
                logger.info("-" * 80)
                logger.info("===== 全流程测试执行完成，核心结果预览 =====")

                chunks = final_state.get("chunks", [])
                chunk_count = len(chunks)
                md_content = final_state.get("md_content", "")[:150]
                has_embedding = all("dense_vector" in chunk and "sparse_vector" in chunk for chunk in chunks) if chunks else False

                logger.info(f"PDF 转 MD 内容预览（前 150 字符）：{md_content}...")
                logger.info(f"文档切分总切片数：{chunk_count}")
                logger.info(f"所有切片是否完成向量化：{'是' if has_embedding else '否'}")
                logger.info(f"最终识别名称：{final_state.get('item_name', '')}")
                logger.info(f"最终状态包含的核心键：{list(final_state.keys())}")
                logger.info("-" * 80)

        except Exception:
            logger.exception("===== 全流程测试运行失败 =====")

    logger.info("===== 导入全流程测试结束 =====")
