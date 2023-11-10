"""
A model worker executes the model.
"""
import sys, os, io


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

from demo.seem.tasks.interactive import interactive_infer_image

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


conf_files = "configs/seem/focalt_unicl_lang_demo.yaml"
opt = load_opt_from_config_files([conf_files])
opt = init_distributed(opt)

def prepare_image(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    if os.path.exists(image_pth):
        image = Image.open(image_pth).convert('RGB')
    else:
        image = Image.open(BytesIO(base64.b64decode(image_pth))).convert("RGB")

    return image


def prepare_mask(image_pth):
    """
    apply transformation to the image. crop the image ot 640 short edge by default
    """
    if os.path.exists(image_pth):
        image = Image.open(image_pth).convert('RGBA')
    else:
        image = Image.open(BytesIO(base64.b64decode(image_pth))).convert("RGBA")

    return image

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

        # load model
        self.model = self.build_model()



        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,)
        )
        self.heart_beat_thread.start()
        

    def build_model(self):
        # META DATA
        cur_model = 'None'
        if 'focalt' in conf_files:
            pretrained_pth = os.path.join("seem_focalt_v0.pt")
            if not os.path.exists(pretrained_pth):
                os.system("wget {}".format(
                    "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
            cur_model = 'Focal-T'
        elif 'focal' in conf_files:
            pretrained_pth = os.path.join("seem_focall_v0.pt")
            if not os.path.exists(pretrained_pth):
                os.system("wget {}".format(
                    "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt"))
            cur_model = 'Focal-L'

        '''
        build model
        '''
        model = BaseModel(opt, build_model(opt)).from_pretrained(
            pretrained_pth).eval().cuda()
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
        return model

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


    def generate_stream_func(self, model_path, model_type, params):
        
        image_path = params["image"]
        pil_image = prepare_image(image_path)
        
        refimg_path = params["refimg"]
        pil_refimg = prepare_image(refimg_path)
        
        refmask_path = params["refmask"]
        pil_refmask = prepare_mask(refmask_path)
        
        image_input = {
            'image': pil_image,
            "mask": None,
        }
        
        refimg_input = {
            'image': pil_refimg,
            "mask": pil_refmask,
        }
        
        res_img = interactive_infer_image(self.model, None, image_input, 'Example', refimg=refimg_input)[0]
        
        # to b64        
        buffered = BytesIO()
        res_img.save(buffered, format="JPEG")
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        
        pred_dict = {
            "edited_image": img_b64_str
        }

        return pred_dict

    def generate_gate(self, params):
        # ret = {"text": "", "error_code": 0}
        # ret = self.generate_stream_func(
        #     self.model_path,
        #     self.model_type,
        #     params
        # )
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
    parser.add_argument("--port", type=int, default=21043)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21043")
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
        ['seem']
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
