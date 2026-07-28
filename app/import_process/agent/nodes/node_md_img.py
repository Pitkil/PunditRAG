import base64
from collections import deque
import mimetypes
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple
from langchain.messages import HumanMessage
from minio.deleteobjects import DeleteObject
from app.clients.minio_utils import get_minio_client
from app.conf.minio_config import minio_config
from app.conf.llm_config import llm_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.llm.llm_util import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_done_task, add_running_task
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
from langchain_core.output_parsers import StrOutputParser


def is_supported_image(filename:str):
     '''
     判断文件是否为MinIO支持的图片格式
     '''
     return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

@step_log("step_1")
def step_1(state:ImportGraphState)->Tuple[str,Path,Path]:
    md_content = state["md_content"]
    md_path = state['md_path']
    
    if not md_path:
        logger.error(f"md_path为空!")
        raise ValueError("md_path为空!")
    
    md_path_obj = Path(md_path)

    if not md_content:
        logger.warning("md_content为空!")
        md_content = md_path_obj.read_text(encoding="utf-8")
        state["md_content"] = md_content

    images_path_obj = md_path_obj.parent / "images"

    return md_content,md_path_obj,images_path_obj

@step_log("step_2")
def step_2(md_content:str,images_path_obj:Path)->List[Tuple[str,str,Tuple[str,str]]]:
    image_context_list = []
    #逐个读取图片文件夹里的每一个文件
    for image_file in images_path_obj.iterdir():
        image_name = image_file.name
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是图片,跳过本次!!")
            continue
        #正则匹配![](图片名)
        rep = re.compile(rf"\!\[.*?\]\(.*?{re.escape(image_name)}.*?\)")
        match_obj = rep.search(md_content)
        if not match_obj:
            logger.warning(f"{image_name}没有在md中使用,跳过本次处理!")
            continue
        # 正确获取起始和结束字符索引
        start, end = match_obj.span()
        #增加上下文
        pre_content = md_content[max(0,start-150):start]
        pos_content = md_content[end:min(end+150,len(md_content))]
        image_context_list.append((image_name,str(image_file),(pre_content,pos_content)))
    return image_context_list

@step_log("step_3")
def step_3(image_context_list, stem) -> Dict[str,str]:
    image_summary_dict = {}

    vl_model = get_llm_client(llm_config.vl_model)

    for image_name,image_path,content in image_context_list:
        apply_api_rate_limit(3,60)
        #构建提示词
        image_path_obj = Path(image_path)
        image_prompt = load_prompt("image_summary",root_folder = stem,image_content = content)
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        message = HumanMessage(
            content = [
    {
          "type": "image_url",
          "image_url": {
            "url":  f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_data}"
          }, 
        },
        {"type": "text", "text": image_prompt},
  ],
        )
        chains = vl_model | StrOutputParser()
        summary = chains.invoke([message])
        image_summary_dict[image_name] = summary

    return image_summary_dict

@step_log("step_4")
def step_4(image_context_list, image_summaries_dict, md_content, stem) -> str:
     minio_client = get_minio_client()
     image_root_dir = str(minio_config.minio_img_dir or "").strip("/")
     image_object_prefix = f"{image_root_dir}/{stem}".strip("/")
     object_list = minio_client.list_objects(
        bucket_name= str(minio_config.bucket_name),
        prefix = f"{image_object_prefix}/",
        recursive=True,
     )

     #删除上次处理残留
     delete_object_list = [DeleteObject(str(obj.object_name))  for obj in object_list]

     errors = minio_client.remove_objects(
        # 参数1: 要删除桶的名字
        bucket_name=str(minio_config.bucket_name),
        delete_object_list=delete_object_list
    )
     
     for error in errors:
        logger.warning(f"删除失败,失败原因:{error}")

     logger.debug("删除成功!")

     # 循环"图片地址"地址向minio服务器传递文件,并且拼接(端点/桶名/对象名)和记录地址 {图片名:图片地址}
     image_url_dict = {}
     for  image_name,image_path, _ in   image_context_list:
         try:
             image_object_name = f"{image_object_prefix}/{image_name}"
             minio_client.fput_object(
                 bucket_name=str(minio_config.bucket_name),
                 object_name=image_object_name,
                 file_path=str(image_path),
                 content_type=str(mimetypes.guess_type(image_name)[0])
             )
             # 拼接图片的网络地址  端点 + 桶 + 对象名
             protocol = "https" if minio_config.minio_secure else "http"
             image_minio_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}/{image_object_name}"
             logger.debug(f"图片:{image_name}上传成功!回显地址:{image_minio_url}")
             # 存储图片和对应网络地址对
             image_url_dict[image_name] = image_minio_url
         except Exception as e:
            logger.warning(f"本次图片上传失败:{image_name},跳过继续上传下一张!")
            continue
      # 4. {图片名:图片地址}   {图片名 : 图片的总结和描述} -> 合并到一起 -> {图片名 : (图片描述和总结,图片地址)}
      # {图片名: (图片描述和总结, 图片地址)}
     total_image_info = {}
     if not image_url_dict:
            logger.warning("图片上传全部失败!")
            return md_content
     
     for image_name,image_url in image_url_dict.items():
         total_image_info[image_name] = (image_url,image_summaries_dict[image_name])

     #替换成 MinIO 网络 URL 后，文档就可以在任意地方渲染预览了。
     for image_name,(image_url,image_summary) in total_image_info.items():
        # 找到 md_content   ![summary](image_url) -> ![](/xxx/xxx.jpg)
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)")
        #lambda：防止反斜杠或数字（比如 \1、\g）被当作正则    （替换后的格式和内容，被替换的内容）
        md_content =  rep.sub(lambda _:f"![{image_summary}]({image_url})",md_content)

     return md_content

@step_log("step_5")
def step_5(new_md_content, md_path_obj) -> str:
    """
    将新的md_content内容写入到本地磁盘! xx.md xx_new.md
    :param new_md_content:
    :param md_path_obj:
    :return: 新md的地址 str
    """
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    # 备份
    new_md_path_obj.write_text(new_md_content,encoding="utf-8")
    return str(new_md_path_obj)

      


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    #开始任务
    add_running_task(state['task_id'],"node_md_img")
    #准备和校验
    md_context,md_path_obj,images_path_obj = step_1(state)
    #提前结束识别
    if not images_path_obj.exists() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning(f"图片文件夹为空或者没有图片,无需后续处理!")
        return state
    #扫描图片上下文
    image_context_list = step_2(md_context,images_path_obj)
    logger.info(f"已经获取上下文信息:{image_context_list}")
    #调用视觉模型识别图片内容
    image_summary_list = step_3(image_context_list,md_path_obj.stem)
    #上传图片并替换md内容
    new_md_content = step_4(image_context_list,image_summary_list,md_context,md_path_obj.stem)
    #备份
    new_md_path = step_5(new_md_content,md_path_obj)
    #更新状态
    state['md_content'] = new_md_content
    state['md_path'] = new_md_path
    add_done_task(state['task_id'],"node_md_img")
    return state
