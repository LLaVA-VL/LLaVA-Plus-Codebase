"""
A model worker executes the model.
Using 
"""
import sys, os
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

from clip_retrieval.clip_client import ClipClient, Modality


import torch
import torch.nn.functional as F
import uvicorn
import clip  # pylint: disable=import-outside-toplevel

from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from serve.utils import build_logger, pretty_print_semaphore

from PIL import Image as pimage


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
        model_path,
        model_names,
        device,
        indice_name,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device

        # load clip
        logger.info(f"Loading clip ...")
        self.model, self.preprocess = clip.load("ViT-L/14", device=device, jit=True)


        # load model
        logger.info(f"Loading client ...")
        self.client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name=indice_name,
            aesthetic_score=9,
            aesthetic_weight=0.5,
            modality=Modality.IMAGE,
            num_images=20,
        )

        if indice_name == "laion5B-H-14":
            os.environ['SEND_IMAGE'] = "True"
            print("Set SEND_IMAGE to True, since indice_name is laion5B-H-14")
        print("SEND_IMAGE", os.getenv("SEND_IMAGE", False))

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def get_text_emb(self, text):
        with torch.no_grad():
            text_emb = self.model.encode_text(clip.tokenize([text], truncate=True).to(self.device))
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.cpu().detach().numpy().astype("float32")[0]
        return text_emb


    def get_image_emb(self, image):
        with torch.no_grad():
            image_emb = self.model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))
            image_emb /= image_emb.norm(dim=-1, keepdim=True)
            image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
            return image_emb

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

    def load_image(self, image_path: str) -> Image.Image:
    
        if os.path.exists(image_path):
            image_source = Image.open(image_path).convert("RGB")
        else:
            # base64 coding
            image_source = Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")

        return image_source

    def generate_stream_func(self, model, params, device):
        # get inputs
        image_path = params["image"]
        # load image and run models
        image = self.load_image(image_path)
        

        if not os.getenv("SEND_IMAGE", False):


            # embedding
            tic = time.time()
            image_emb = self.get_image_emb(image)
            embed_time = time.time() - tic

            # search
            tic = time.time()
            beach_results = self.client.query(embedding_input=image_emb.tolist())
            query_time = time.time() - tic

            print("embed_time", embed_time, "query_time", query_time)
            print(beach_results)
        else:
            # save image
            save_name = "test.jpg"
            image.save(save_name)
            beach_results = self.client.query(image=save_name)
            

        w, h = image.size
        pred_dict = {
            "retrieval_results": beach_results,
            "size": [h, w],  # H,W
        }

        return pred_dict

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



# app = FastAPI()


# def release_model_semaphore():
#     model_semaphore.release()


# def acquire_model_semaphore():
#     global model_semaphore, global_counter
#     global_counter += 1
#     if model_semaphore is None:
#         model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
#     return model_semaphore.acquire()


# def create_background_tasks():
#     background_tasks = BackgroundTasks()
#     background_tasks.add_task(release_model_semaphore)
#     return background_tasks



# @app.post("/worker_generate")
# async def api_generate(request: Request):
#     params = await request.json()
#     await acquire_model_semaphore()
#     output = worker.generate_gate(params)
#     release_model_semaphore()
#     return JSONResponse(output)


# @app.post("/worker_get_status")
# async def api_get_status(request: Request):
#     return worker.get_status()





# @app.post("/model_details")
# async def model_details(request: Request):
#     return {"context_length": worker.context_len}


if __name__ == "__main__":
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


    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21046)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21046")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )

    parser.add_argument(
        "--model-path", type=str, default="Salesforce/blip2-opt-2.7b"
    )

    parser.add_argument(
        "--model-names",
        default="retrieval,clip",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )

    parser.add_argument(
        "--indice_name",
        type=str,
        default="laion5B-H-14",
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
        args.model_path,
        args.model_names,
        args.device,
        args.indice_name,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
