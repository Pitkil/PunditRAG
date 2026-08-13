from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 全局客户端单例
_milvus_client = None


def get_milvus_client():
    """
    Milvus客户端单例获取，复用连接避免重复创建
    :return: MilvusClient实例，连接失败返回None
    """
    try:
        global _milvus_client
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    将chunk_id转换为INT64（主键schema为INT64），过滤无效ID
    :param ids: chunk_id列表
    :return: (ok_ids, bad_ids)，可转换/不可转换的ID列表
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    通过chunk_id主键批量查询切片数据
    优先用get方法（性能最优），失败回退query过滤查询
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param chunk_ids: chunk_id列表
    :param output_fields: 返回字段，默认核心切片字段
    :param batch_size: 分批大小，默认100
    :return: List[dict]，失败返回空列表
    """
    if client is None or not collection_name:
        return []
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")
    if not ok_ids:
        return []

    results = []
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # 优先主键get查询
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        # 回退filter过滤查询
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建混合搜索请求对象，分别创建稠密/稀疏向量搜索请求
    :param dense_vector: 稠密向量
    :param sparse_vector: 稀疏向量
    :param dense_params: 稠密搜索参数，默认IP（与建库一致）
    :param sparse_params: 稀疏搜索参数，默认IP
    :param expr: 过滤表达式
    :param limit: 单向量返回数量，默认5
    :return: [dense_req, sparse_req]
    """
    if dense_params is None:
        dense_params = {"metric_type": "IP"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行稠密+稀疏向量混合搜索，基于WeightedRanker加权融合
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表[dense_req, sparse_req]
    :param ranker_weights: 融合权重，默认(0.5,0.5)
    :param norm_score: 是否归一化评分后再融合
    :param limit: 返回结果数量，默认5
    :param output_fields: 返回字段，默认item_name
    :param search_params: 搜索参数
    :return: 混合搜索结果列表，失败返回None
    """
    try:
        #待会两边搜出来的结果，左边乘以 0.5，右边乘以 0.5，最后加起来算总分
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        #如果调用者没指定要返回什么数据，默认只把item_name从数据库里打包带回来
        if output_fields is None:
            output_fields = ["item_name"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )
        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None