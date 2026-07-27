import json

from minio import Minio

from app.conf.minio_config import minio_config
from app.core.logger import logger


minio_client = None


def create_minio_client():
    client = Minio(
        endpoint=str(minio_config.endpoint),
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.minio_secure,
    )
    return client


def create_minio_bucket(client: Minio):
    bucket_name = str(minio_config.bucket_name)
    if not client.bucket_exists(bucket_name):
        logger.info(f"MinIO存储桶[{bucket_name}]不存在，开始创建")
        client.make_bucket(bucket_name)
        logger.info(f"MinIO存储桶[{bucket_name}]创建成功")
    else:
        logger.info(f"MinIO存储桶[{bucket_name}]已存在，无需重复创建")

    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }
    client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
    logger.info(f"MinIO存储桶[{bucket_name}]公开读策略已设置")


def get_minio_client():
    global minio_client

    if not minio_client:
        logger.info("开始初始化MinIO客户端")
        client = create_minio_client()
        create_minio_bucket(client)
        minio_client = client
        logger.info("MinIO客户端初始化完成，已就绪可使用")

    return minio_client
