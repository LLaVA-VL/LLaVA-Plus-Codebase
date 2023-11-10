"""
A model worker executes the model.
"""
import sys, os
from groundingdino.util import box_ops

from segment_anything import build_sam
from segment_anything.predictor import SamPredictor
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

from io import BytesIO
import base64

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import requests
from PIL import Image

from demo.inference_on_a_image import get_grounding_output

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
import pycocotools.mask as mask_util


try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch
import torch.nn.functional as F
import uvicorn

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


class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        no_register,
        grounding_dino_server,
        blip2_server,
        model_names,
        device,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device

        self.grounding_dino_server = grounding_dino_server
        self.grounding_dino_server_addr = self.get_sub_worker_addr(grounding_dino_server)
        print(f"grounding_dino_server_addr: {self.grounding_dino_server_addr}")

        self.blip2_server = blip2_server
        self.blip2_server_addr = self.get_sub_worker_addr(blip2_server)
        print(f"blip2_server_addr: {self.blip2_server_addr}")


        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()


    def get_sub_worker_addr(self, worker_name):
        # get grounding dino addr
        if worker_name.startswith("http"):
            sub_server_addr = worker_name
        else:
            controller_addr = self.controller_addr
            ret = requests.post(controller_addr + "/refresh_all_workers")
            ret = requests.post(controller_addr + "/list_models")
            models = ret.json()["models"]
            models.sort()
            print(f"Models: {models}")

            ret = requests.post(
                controller_addr + "/get_worker_address", json={"model": worker_name}
            )
            sub_server_addr = ret.json()["address"]
        # print(f"worker_name: {worker_name}")
        return sub_server_addr

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

    def load_image(self, image_path: str) -> Tuple[np.array, torch.Tensor]:
        

        if os.path.exists(image_path):
            image_source = Image.open(image_path).convert("RGB")
        else:
            # base64 coding
            image_source = Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")

        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def generate_stream_func(self, model, params, device):
        # get inputs


    
        headers = {"User-Agent": "G-SAM Client"}

        # blip2
        pred_dict_blip2 = requests.post(
            self.blip2_server_addr + "/worker_generate",
            headers=headers,
            json=params,
        ).json()

        params.update({
            "caption": pred_dict_blip2["caption"],
        })

        # gdino
        pred_dict_gdino = requests.post(
            self.grounding_dino_server_addr + "/worker_generate",
            headers=headers,
            json=params,
        ).json()

        pred_dict_gdino.update({
            "caption": pred_dict_blip2["caption"],
        })

        return pred_dict_gdino

    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(
                None,
                params,
                self.device,
            )
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
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
    parser.add_argument("--port", type=int, default=21078)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21078")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )

    parser.add_argument(
        "--grounding-dino-server", type=str, default="grounding_dino"
    )
    parser.add_argument(
        "--blip2-server", type=str, default="blip2"
    )
    parser.add_argument(
        "--model-names",
        default="blip2+grounding_dino",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.grounding_dino_server,
        args.blip2_server,
        args.model_names,
        args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
