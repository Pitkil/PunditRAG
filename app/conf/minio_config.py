from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


# 定义MinIO对象存储服务配置
@dataclass
class MinIOConfig:
    endpoint: Optional[str]    # MinIO服务地址（含http/https和端口）
    public_endpoint: Optional[str] # 浏览器可访问的MinIO地址
    access_key: Optional[str]  # MinIO访问密钥（对应MINIO_ACCESS_KEY）
    secret_key: Optional[str]  # MinIO秘钥（对应MINIO_SECRET_KEY）
    bucket_name: Optional[str] # MinIO默认存储桶名（知识库文件专用）
    minio_img_dir: Optional[str] #Minio存储图片的文件夹
    minio_secure: bool # 是否使用ssl加密 http 还是 https
    public_read: bool
    asset_base_url: str


# 实例化MinIO配置对象，自动从.env读取配置并绑定
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    public_endpoint=os.getenv("MINIO_PUBLIC_ENDPOINT") or os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=(os.getenv("MINIO_SECURE") or "False") == "True",
    public_read=(os.getenv("MINIO_PUBLIC_READ") or "False").lower() == "true",
    asset_base_url=(os.getenv("ASSET_BASE_URL") or "http://127.0.0.1:8000/assets").rstrip("/"),
)
