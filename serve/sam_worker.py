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
        model_names,
        sam_path,
        device,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()


        # load sam model
        self.sam = build_sam(checkpoint=sam_path)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)

    

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
        # image_transformed, _ = self.transform(image_source, None)
        return image, None

    def generate_stream_func(self, model, params, device):
        # get inputs
        image_path = params["image"]
        
        boxes = params.get("boxes", None)                   # b, 4
        points = params.get("points", None)                 # b, n, 2
        point_labels = params.get("point_labels", None)     # b, n. (1 indicates a foreground point and 0 indicates a background point.) 

        assert not (boxes is None and points is None), "boxes and points cannot be both None"
        assert not (boxes is not None and points is not None), "boxes and points cannot be both not None"

        image_np, _ = self.load_image(image_path)
        h, w, _ = image_np.shape

        # add sam output
        if boxes is not None:
            if len(boxes) > 0:
                boxes_tensor = torch.Tensor(boxes).to(device)
                boxes_xyxy = boxes_tensor * torch.Tensor([w, h, w, h]).to(device)
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
                maskrls_list = []
        elif points is not None:
            assert point_labels is not None, "point_labels cannot be None when points is not None"
            if len(points) > 0:
                points_tensor = torch.Tensor(points).to(device) * torch.Tensor([w, h]).to(device)
                self.sam_predictor.set_image(image_np)
                transformed_points = self.sam_predictor.transform.apply_coords_torch(points_tensor, image_np.shape[:2]).to(device)

                point_labels_tensor = torch.Tensor(point_labels).to(device)
                # import ipdb; ipdb.set_trace()
                masks, _, _ = self.sam_predictor.predict_torch(
                            point_coords = transformed_points, # b, n, 2
                            point_labels = point_labels_tensor, # b, n. 1 indicates a foreground point and 0 indicates a background point. 
                            boxes = None,
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
                maskrls_list = []            
        
        pred_dict = {}
        pred_dict['masks_rle'] = maskrls_list
        pred_dict['boxes'] = boxes
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

    @torch.inference_mode()
    def get_embeddings(self, params):
        try:
            tokenizer = self.tokenizer
            is_llama = "llama" in str(
                type(self.model)
            )  # vicuna support batch inference
            is_chatglm = "chatglm" in str(type(self.model))
            is_t5 = "t5" in str(type(self.model))
            if is_llama:
                encoding = tokenizer.batch_encode_plus(
                    params["input"], padding=True, return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                model_output = self.model(
                    input_ids, attention_mask, output_hidden_states=True
                )
                data = model_output.hidden_states[-1]
                mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
                masked_embeddings = data * mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                seq_length = torch.sum(mask, dim=1)
                embedding = sum_embeddings / seq_length
                normalized_embeddings = F.normalize(embedding, p=2, dim=1)
                ret = {
                    "embedding": normalized_embeddings.tolist(),
                    "token_num": torch.sum(attention_mask).item(),
                }
            else:
                embedding = []
                token_num = 0
                for text in params["input"]:
                    input_ids = tokenizer.encode(text, return_tensors="pt").to(
                        self.device
                    )
                    if is_t5:
                        model_output = self.model(
                            input_ids, decoder_input_ids=input_ids
                        )
                    else:
                        model_output = self.model(input_ids, output_hidden_states=True)
                    if is_chatglm:
                        data = (model_output.hidden_states[-1].transpose(0, 1))[0]
                    elif is_t5:
                        data = model_output.encoder_last_hidden_state[0]
                    else:
                        data = model_output.hidden_states[-1][0]
                    data = F.normalize(torch.mean(data, dim=0), p=2, dim=0)
                    embedding.append(data.tolist())
                    token_num += len(input_ids[0])
                ret = {
                    "embedding": embedding,
                    "token_num": token_num,
                }
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
    parser.add_argument("--port", type=int, default=21273)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21273")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )

    parser.add_argument(
        "--sam-path", type=str, default="sam_vit_h_4b8939.pth"
    )
    parser.add_argument(
        "--model-names",
        default="sam",
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
        args.model_names,
        args.sam_path,
        args.device,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
