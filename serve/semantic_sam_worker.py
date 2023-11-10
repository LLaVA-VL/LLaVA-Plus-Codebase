"""
A model worker executes the model.
"""
import sys, os, io

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from semantic_sam import plot_multi_results, build_semantic_sam, SemanticSAMPredictor

import argparse
import asyncio
import dataclasses
import logging
import json
import os
import sys
import time
from typing import List, Tuple, Union
import threading
import uuid
import torchvision
import numpy as np
from io import BytesIO
import base64

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import uvicorn
from torchvision import transforms


from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from serve.utils import build_logger, pretty_print_semaphore

GB = 1 << 30


now_file_name = os.__file__
logdir = "logs/workers/"
os.makedirs(logdir, exist_ok=True)
logfile = os.path.join(logdir, f"{now_file_name}.log")

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(now_file_name, logfile)
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def prepare_image(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    if os.path.exists(image_pth):
        image = Image.open(image_pth).convert('RGB')
    else:
        image = Image.open(BytesIO(base64.b64decode(image_pth))).convert("RGB")
    t = []
    t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
    transform1 = transforms.Compose(t)
    image_ori = transform1(image)

    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()

    return image_ori, images

class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        model_path,
        model_type,
        model_names
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]


        logger.info(f"Loading the model on worker {worker_id} ...")
        self.model_path = model_path
        self.model_type = model_type
        self.model_names = model_names

        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,)
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }


    def generate_stream_func(self,model_path,model_type, params):
        # get inputs
        if 'point' in params:
            point_prompt = params["point"]
        else:
            assert 'boxes' in params
            box_prompt = params["boxes"]
            # centers
            point_prompt = []
            for box in box_prompt:
                point_prompt.append([(box[0]+box[2])/2, (box[1]+box[3])/2])
        image_path = params["image"]
        model_type = model_type
        model_ckpt = model_path
        # model_name = params["model"]

        # load image and run models
        original_image, input_image = prepare_image(image_path)  # change the image path to your image
        mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type=model_type, ckpt=model_ckpt)) # model_type: 'L' / 'T', depends on your checkpint
        iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point=point_prompt) # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
        
        # 创建一个BytesIO对象，用于临时存储图像数据
        image_data = io.BytesIO()
        sum_0=np.array(iou_sort_masks[0]).sum()
        sum_1=np.array(iou_sort_masks[1]).sum()
        mask_List = []
        # 将图像保存到BytesIO对象中，格式为JPEG
        for mask in iou_sort_masks :
            image_data = io.BytesIO()
            mask.save(image_data, format='JPEG')
            # 将BytesIO对象的内容转换为字节串
            image_data_bytes = image_data.getvalue()
            # 将图像数据编码为Base64字符串
            encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
            
            mask_List.append(encoded_image)
        
        pred_dict = {
            "iou_sort_masks": mask_List
        }

        return pred_dict

    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(
                self.model_path,
                self.model_type,
                params
            )
        # except torch.cuda.OutOfMemoryError as e:
        #     ret = {
        #         "text": f"{SERVER_ERROR_MSG}\n\n({e})",
        #         "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
        #     }
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


app = FastAPI()


def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks



@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    output = worker.generate_gate(params)
    release_model_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()





@app.post("/model_details")
async def model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21004)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21004")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )

    parser.add_argument(
        "--model-path", type=str, default="/comp_robot/lifeng/code/Semantic-SAM-worker/ckp/swint_only_sam_many2many.pth"
    )
    parser.add_argument(
        "--model-type", type=str, default="T"
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_type,
        ['semantic_sam', 'semantic-sam']
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
