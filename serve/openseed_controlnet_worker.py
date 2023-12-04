"""
A model worker executes the model.
"""
import numpy as np
np.random.seed(1)
import sys, os, io
# import pdb;pdb.set_trace()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from torchvision import transforms


from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.arguments import load_opt_from_config_files
import argparse
import asyncio

import sys
import time
import threading
import uuid
import base64

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

import torch.nn.functional as F
import uvicorn

from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
from serve.utils_ import build_logger, pretty_print_semaphore

from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np

ade20k_classes=['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window ', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'tub', 'rail', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'pool', 'stool', 'barrel', 'basket', 'falls', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])



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


conf_files="configs/openseed/openseed_swinl_lang_decouple.yaml"
pretrained_pth="/path/to/openseed_ade20k_swinl.pt"
class ModelWorker:
    def __init__(
        self,
        controller_addr,
        worker_addr,
        worker_id,
        model_names,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        
        # self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        # self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float32
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float32
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,)
        )
        self.heart_beat_thread.start()
        opt = load_opt_from_config_files([conf_files])
        # if cmdline_args.user_dir:
        #     absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        #     opt['user_dir'] = absolute_user_dir

        # META DATA
        # pretrained_pth = os.path.join(opt['WEIGHT'])
        # output_root = './output'
        # image_pth = 'images/animals.png'

        self.openseed_model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()


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


    def generate_stream_func(self, params ):
        # get inputs
        image_b64 = params["image"]
        mode = params["mode"]
        if mode=="openseed":
            t = []
            t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
            transform = transforms.Compose(t)

            stuff_classes = ade20k_classes
            stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in
                            range(len(stuff_classes))]
            stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(stuff_classes))}

            MetadataCatalog.get("demo").set(
                stuff_colors=stuff_colors,
                stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            )
            model=self.openseed_model
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
            metadata = MetadataCatalog.get('inference')
            model.model.metadata = metadata
            model.model.sem_seg_head.num_classes = len(stuff_classes)
            with torch.no_grad():
                image_ori = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()

                batch_inputs = [{'image': images, 'height': height, 'width': width}]
                outputs = model.forward(batch_inputs, inference_task="sem_seg")
                # outputs[-1]['sem_seg'] = F.interpolate(outputs[-1]['sem_seg'][None], size=(512, 512), mode='bilinear',
                #                                        align_corners=False)[0].cpu()
                seg = outputs[-1]['sem_seg'].max(0)[1].cpu()

            #seg: 512x512 (0~149)
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color

            color_seg = color_seg.astype(np.uint8)

            image = Image.fromarray(color_seg)
            image_data = io.BytesIO()
            image.save(image_data, format='JPEG')
            image_data_bytes = image_data.getvalue()
            # 将图像数据编码为Base64字符串
            encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
            pred_dict = {
                "image_seg": encoded_image,
            }
        else:
            assert mode=="controlnet"
            text_prompt = params["prompt"]["caption"]
            image=params['mask']
            image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            image1 = self.pipe(text_prompt, image, num_inference_steps=20).images[0]


            image_data1 = io.BytesIO()
            image1.save(image_data1, format='JPEG')
            image_data_bytes1 = image_data1.getvalue()
            # 将图像数据编码为Base64字符串
            encoded_image1 = base64.b64encode(image_data_bytes1).decode('utf-8')
         
            pred_dict = {
                "edited_image": encoded_image1,
            }

        return pred_dict
    
    
    def generate_gate(self, params):
        try:

            ret = {"text": "", "error_code": 0}
            ret = self.generate_stream_func(            
                params
            )
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
    parser.add_argument("--port", type=int, default=22003)
    parser.add_argument("--worker-address", type=str, default="http://localhost:22003")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )

    parser.add_argument(
        "--model-names",
        default="controlnet",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")


    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_names
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
