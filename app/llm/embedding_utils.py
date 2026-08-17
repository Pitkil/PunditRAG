from pathlib import Path
from threading import Lock

from huggingface_hub import snapshot_download
import numpy as np
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
bge_m3_ef = None
_bge_m3_lock = Lock()
# FlagEmbedding mutates the shared model when FP16 encoding starts. Serialize
# inference so concurrent query branches cannot race on model dtype.
_bge_m3_encode_lock = Lock()

def get_bge_m3_ef():
    global bge_m3_ef
    if bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return bge_m3_ef

    with _bge_m3_lock:
        if bge_m3_ef is not None:
            return bge_m3_ef

        local_path = embedding_config.bge_m3_path.strip()
        repository_id = embedding_config.bge_m3.strip()
        local_dir = Path(local_path) if local_path else None
        local_weights = (
            local_dir
            and local_dir.is_dir()
            and any((local_dir / name).is_file() for name in ("pytorch_model.bin", "model.safetensors"))
        )
        if local_dir and local_dir.is_dir() and any(local_dir.iterdir()) and not local_weights:
            raise RuntimeError(
                f"BGE-M3 本地模型尚未下载完整：{local_path}。请等待模型下载完成后重试。"
            )
        model_name = local_path if local_weights else repository_id
        if not model_name:
            raise ValueError("BGE-M3模型未配置，请设置 BGE_M3 或提供有效的 BGE_M3_PATH")
        if local_path and model_name == repository_id:
            logger.warning(f"BGE-M3本地目录为空或不存在：{local_path}，改用模型仓库：{repository_id}")
        if model_name == repository_id:
            logger.info("正在检查BGE-M3模型缓存，首次使用时会自动下载必要文件")
            model_name = snapshot_download(
                repo_id=repository_id,
                ignore_patterns=[
                    "onnx/*",
                    "openvino/*",
                    "*.onnx",
                    "*.jpg",
                    "*.md",
                ],
            )
        device = embedding_config.bge_device
        use_fp16 = embedding_config.bge_fp16

        logger.info(
            "开始初始化BGE-M3模型",
            extra={
                "model_name": model_name,
                "device": device,
                "use_fp16": use_fp16,
                "normalize_embeddings": True
            }
        )

        try:
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                normalize_embeddings=True
            )
            logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
            return bge_m3_ef
        except Exception as e:
            logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
            raise


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # The query graph runs the original and HyDE branches concurrently.
        # FlagEmbedding's FP16 path calls model.half() inside encode_documents,
        # so the shared model must be encoded under one process-local lock.
        with _bge_m3_encode_lock:
            embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引
            # indices（列索引）
            #indptr[i] 告诉你第 i 句话的向量数据从第几个位置开始
            #indptr[i+1] 告诉你到哪里结束
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].astype(np.float32, copy=False).tolist()
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表
        result = {
            "dense": [np.asarray(emb, dtype=np.float32).tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理
