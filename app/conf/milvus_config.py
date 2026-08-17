from dataclasses import dataclass
import os
from dotenv import load_dotenv


load_dotenv()


# 定义Milvus向量数据库配置类
@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus服务端连接地址
    chunks_collection: str   # 存储切片的集合名称
    item_name_collection: str    # 存储文档对应实体类的集合名称

# 实例化Milvus配置对象
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL", ""),
    chunks_collection=os.getenv("CHUNKS_COLLECTION", ""),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION", "")
)
