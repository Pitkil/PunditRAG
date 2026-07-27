from pathlib import Path
import shutil
import sys
import time
from typing import Tuple
from app.conf.mineru_config import mineru_config
from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_done_task, add_running_task
import requests


@step_log("step_1")#装饰器,自动打印日志
def step_1(state:ImportGraphState)->Tuple[Path,Path]:
     """
      入参: state
      出参: pdf_path_obj [Path]  local_dir_obj [Path]

      步骤:
         1. state获取对应的地址
         2. 进行非空校验(pdf_path -> none -> 结束 | local_dir 给与默认地址)
         3. 将两个参数转成Path (str -> Path )
         4. 判断pdf_path_obj是否有文件,local_dir_path 是否存在文件夹
            没有文件->抛出异常
            没有文件夹 -> 创建文件mkdir
         5. 返回两个路径地址
    """
     pdf_path = state["local_file_path"] #PDF 文件的路径。
     local_dir = state["local_dir"] #生成的 Markdown 文件的本地存放目录。

     #判断是否为空
     if not pdf_path:
          logger.error("pdf_path值为空!")
          raise ValueError("pdf_path值为空!")
     if not local_dir:
          logger.warning("没有指定local_dir地址，采取默认值")
          local_dir = PROJECT_ROOT / "output"
          state["local_dir"] = str(local_dir)

     #转换为Path对象
     pdf_path_obj = Path(pdf_path)
     local_dir_obj = Path(local_dir)

     #判断是否存在
     if not pdf_path_obj.exists():
          logger.error(f"pdf_path:{pdf_path_obj}，没有文件存在！")
          raise FileNotFoundError(f"pdf_path:{pdf_path_obj}，没有文件存在！")

     if not local_dir_obj.exists():
               logger.warning(f"local_path:{local_dir_obj}，没有文件夹存在，主动创建！")
               # parents=True  可以创建多层结构文件夹
               # exist_ok=True 存在也不报错! 没有才会创建
               local_dir_obj.mkdir(parents=True,exist_ok=True)
     return pdf_path_obj,local_dir_obj

@step_log("step_2")
def step_2(pdf_path_obj) -> str: 
      """
       入参: pdf_path_obj
       出参: zip_url (str)

       步骤:
         1. 参数校验 (minerU -> 检查下miner url和key)
         2. 申请上传地址 (minerU) [batch_id]
         3. 向执行地址进行上传文件
         4. 轮询获取返回结果(zip_url) [batch_id]
         5. 返回zip_url
       """
      if not mineru_config.base_url or not mineru_config.api_key:
        logger.error(f"minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

      token = mineru_config.api_key
      url = f"{mineru_config.base_url}/file-urls/batch"
      header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
      data = { 
            "files": [
                {"name":f"{pdf_path_obj.stem}.pdf"}## 获取剥离后缀后的 PDF 纯文件名（例如 'C:/docs/report.pdf' -> 'report'）
            ],
            "model_version":"vlm"
        }
      response = requests.post(url, headers=header, json=data, timeout=30)
      http_status_code = response.status_code
      if http_status_code != 200:
        logger.error(f"申请上传地址失败,返回状态码为:{http_status_code},请检查minerU配置!")
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{http_status_code},请检查minerU配置!")

      result_dict = response.json()
      if result_dict['code'] != 0:
        
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}")
        raise RuntimeError(f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}")

      file_upload_url = result_dict['data']['file_urls'][0]
      batch_id =  result_dict['data']['batch_id']

      data = pdf_path_obj.read_bytes()

       # 获取session对象，以得到纯净版的请求头,不随意携带代理的参数
      with requests.Session() as session:
           session.trust_env = False
           upload_response = session.put(file_upload_url, data=data, timeout=60)
           # http状态码的含义: 1xx 中继状态 (请求没有完成)  2xx 成功状态(200 完整请求成功 202 端点续传成功)  3xx 重定向状态(302 304 支付宝支付)
           # 4xx 客户端错误(404 405 400) [前端]  5xx 服务器错误(500 502 504) [后端]
           if upload_response.status_code != 200:
                logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
                raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")

      #轮询获取结果
      poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
      timeout = 600 #秒 1页pdf 0.5-1秒左右
      interval = 3 #轮询间隔时间
      start_time = time.time()
      while True:
           if time.time() - start_time > timeout:
                logger.error(f"轮询超时,请检查minerU配置!")
                raise TimeoutError(f"轮询超时,请检查minerU配置!")
           try:
                poll_response = requests.get(poll_url, headers=header, timeout=30)
           except Exception as e:
                logger.warning(f"请求出现异常!稍后重试!")
                time.sleep(interval)
                continue

           http_poll_status_code = poll_response.status_code

           #判断网络请求状态码
           if http_poll_status_code !=200:
                if  500 <= http_poll_status_code < 600:
                    # 给机会
                    logger.warning(f"可有修复的网络异常,状态码为:{http_poll_status_code}")
                    time.sleep(interval)
                    continue
                else:
                    logger.error(f"不可修复的网络状态异常,状态码为:{http_poll_status_code}")
                    raise RuntimeError(f"不可修复的网络状态异常,状态码为:{http_poll_status_code}")

           # 判断业务状态码
           poll_response_dict = poll_response.json()
           if poll_response_dict['code'] != 0:
                logger.error(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
                raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")

           # 判断具体的转化状态 state
           extract_result = poll_response_dict['data']['extract_result'][0]
           extract_result_state = extract_result['state']

           if extract_result_state == 'done':
                extract_result_url = extract_result['full_zip_url']
                if not extract_result_url:
                     logger.error(f"已经完成了解析,但是zip地址为空!")
                     raise RuntimeError(f"已经完成了解析,但是zip地址为空!")
                # 返回压缩地址
                return extract_result_url
           elif extract_result_state == 'failed':
                logger.error(f"解析失败了!失败信息:{extract_result['err_msg']}")
                raise RuntimeError(f"解析失败了!失败信息:{extract_result['err_msg']}")
           else:
                logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
                time.sleep(interval)
                continue

@step_log("step_3")
def step_3(zip_url, local_dir_path_obj, stem) -> Path:
    """
     1. 向指定zip地址发起请求获取响应response
     2. 将响应数据写到本地磁盘 [local_dir_path/pdf_path_obj.stem/stem.zip]
     3. 先清空解压文件夹的原文件
     4. 再次解压即可
     5. 检查是否存在md文件
     6. 进行md文件的命名确定 [xx.pdf -> full.md -> xx.md]
     7. 返回md_path_obj地址
    """
    # 向指定地址发起请求获取下载的文件内容zip
    response = requests.get(zip_url,timeout=30)
    # 将下载内容写到本地
    md_path_obj = local_dir_path_obj / f"{stem}_result.zip"
    md_path_obj.write_bytes(response.content)
    #  准备解压对应的文件夹
    extract_path_obj = local_dir_path_obj / stem 
    #删除上次的残留文件
    if extract_path_obj.exists():
        shutil.rmtree(extract_path_obj)
    # 创建好对应的文件
    extract_path_obj.mkdir(parents=True,exist_ok=True)
    #解压
    shutil.unpack_archive(md_path_obj,extract_path_obj)
    #找.md文件
    md_file_list = list(extract_path_obj.rglob("*.md"))
    if not md_file_list or len(md_file_list) == 0:
        logger.error(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")
        raise FileNotFoundError(f"文件解压失败,在:{extract_path_obj}没有任何md文件!")
    # minerU 命名不确定 
    #第一种，名字没变，直接返回
    target_md_obj = None
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_obj = md_file
            return target_md_obj
    # 第二种，full.md
    for md_file in md_file_list:
        if md_file.name.lower() == "full.md":
            target_md_obj = md_file
            break
    # 随机.md
    if not target_md_obj:
        target_md_obj = md_file_list[0]

    # 统一修改文件名称
    final_md_path_obj = target_md_obj.rename(target_md_obj.with_name(f"{stem}.md"))
    return final_md_path_obj


      
     

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
     # 加入任务
    add_running_task(state["task_id"],"node_pdf_to_md")

    # 校验pdf和输出地址
    pdf_path_obj ,local_dir_path_obj = step_1(state)

    #  像minerU请求+解析
    zip_url = step_2(pdf_path_obj)
    logger.info(f"minerU返回的zip地址:{zip_url}")

    # 4. step_3 下载提取和解压
    md_path_obj =step_3(zip_url,local_dir_path_obj,pdf_path_obj.stem)

    # 根据md地址读取对应md_content内容,并且更新state
    state['md_path'] = str(md_path_obj)
    md_content = md_path_obj.read_text(encoding="utf-8")
    state['md_content'] = md_content

    # 任务完成
    add_done_task(state['task_id'],"node_pdf_to_md")

    return state

