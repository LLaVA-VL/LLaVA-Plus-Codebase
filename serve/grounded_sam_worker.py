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
        model_path,
        model_config,
        model_names,
        sam_path,
        device,
        grounding_dino_server=None,
        sam_server=None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names
        self.model_config = model_config
        self.device = device
        self.grounding_dino_server = grounding_dino_server

        if grounding_dino_server is None:
            raise NotImplementedError("grounding_dino_server is None, we only support grounding_dino_server now.")
            logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
            self.model = load_model(
                model_config_path=model_config,
                model_checkpoint_path=model_path,
                device=device,
            )
            self.model.to(device)
            self.model.eval()
        else:
            self.model = None
            # get grounding dino addr
            if grounding_dino_server.startswith("http"):
                grounding_dino_server_addr = grounding_dino_server
            else:
                controller_addr = self.controller_addr
                ret = requests.post(controller_addr + "/refresh_all_workers")
                ret = requests.post(controller_addr + "/list_models")
                models = ret.json()["models"]
                models.sort()
                print(f"Models: {models}")

                ret = requests.post(
                    controller_addr + "/get_worker_address", json={"model": grounding_dino_server}
                )
                grounding_dino_server_addr = ret.json()["address"]
            print(f"grounding_dino_server_addr: {grounding_dino_server_addr}")
            self.grounding_dino_server_addr = grounding_dino_server_addr


        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # load sam model
        self.sam_server = sam_server
        if sam_server is None:
            self.sam = build_sam(checkpoint=sam_path)
            self.sam.to(device=device)
            self.sam_predictor = SamPredictor(self.sam)
            self.sam_server_addr = None
        else:
            self.sam = None
            self.sam_predictor = None
            # get grounding dino addr
            if sam_server.startswith("http"):
                sam_server_addr = sam_server
            else:
                time.sleep(3)
                controller_addr = self.controller_addr
                ret = requests.post(controller_addr + "/refresh_all_workers")
                ret = requests.post(controller_addr + "/list_models")
                models = ret.json()["models"]
                models.sort()
                print(f"Models: {models}")

                ret = requests.post(
                    controller_addr + "/get_worker_address", json={"model": sam_server}
                )
                sam_server_addr = ret.json()["address"]
            print(f"sam_server_addr: {sam_server_addr}")
            self.sam_server_addr = sam_server_addr

    

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
        text_prompt = params["caption"]
        image_path = params["image"]
        box_threshold = params["box_threshold"]
        text_threshold = params["text_threshold"]

        image_np, image = self.load_image(image_path)
        if self.grounding_dino_server is not None:
            headers = {"User-Agent": "G-SAM Client"}
            pred_dict = requests.post(
                self.grounding_dino_server_addr + "/worker_generate",
                headers=headers,
                json=params,
            ).json()
            boxes = pred_dict["boxes"]
            logits = pred_dict["logits"]
            phrases = pred_dict["phrases"]
            h, w = pred_dict["size"]
        else:
            # load image and run models
            boxes, logits, phrases = predict(
                model=model, 
                image=image, 
                caption=text_prompt, 
                box_threshold=box_threshold, 
                text_threshold=text_threshold,
                device=device
            )
            boxes = boxes.tolist()
            # round to 2 decimal places
            boxes = [[round(x, 2) for x in box] for box in boxes]
            logits = logits.tolist()
            logits = [round(x, 2) for x in logits]

            h, w, _ = image_np.shape
            pred_dict = {
                "boxes": boxes,
                "logits": logits,
                "phrases": phrases,
                "size": [h, w],  # H,W
            }

        # add sam output
        if len(boxes) > 0:
            if self.sam_server_addr is None:
                boxes_tensor = torch.Tensor(boxes).to(device)
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_tensor) * torch.Tensor([w, h, w, h]).to(device)
                self.sam_predictor.set_image(image_np)
                transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(device)
                # import ipdb; ipdb.set_trace()
                masks, _, _ = self.sam_predictor.predict_torch(
                            point_coords = None,
                            point_labels = None,
                            boxes = transformed_boxes,
                            multimask_output = False,
                        )
                masks = masks[:, 0] # B, H, W

                # encoder masks to strs
                maskrls_list = []
                for mask in masks:
                    mask_rle = mask_util.encode(np.array(mask[:, :, None].cpu(), order="F"))[0]
                    mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
                    maskrls_list.append(mask_rle)
            else:
                headers = {"User-Agent": "G-SAM Client"}
                params['boxes'] = boxes
                pred_dict_sam = requests.post(
                    self.sam_server_addr + "/worker_generate",
                    headers=headers,
                    json=params,
                ).json()
                maskrls_list = pred_dict_sam['masks_rle']
        else:
            maskrls_list = []
        

        pred_dict['masks_rle'] = maskrls_list
        return pred_dict

    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(
                self.model,
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
    parser.add_argument("--port", type=int, default=21293)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21293")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )

    parser.add_argument(
        "--model-path", type=str, default="groundingdino_swint_ogc.pth"
    )
    parser.add_argument(
        "--model-config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    parser.add_argument(
        "--sam-path", type=str, default="sam_vit_h_4b8939.pth"
    )
    parser.add_argument(
        "--model-names",
        default="grounding_dino+sam,grounded_sam",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--grounding-dino-server", type=str, default="grounding_dino")
    parser.add_argument("--sam-server", type=str, default="sam")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_config,
        args.model_names,
        args.sam_path,
        args.device,
        args.grounding_dino_server,
        args.sam_server,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
