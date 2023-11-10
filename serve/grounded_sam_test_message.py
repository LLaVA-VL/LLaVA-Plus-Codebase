"""Send a test message."""
import argparse
import json
import time
from io import BytesIO
import cv2
from groundingdino.util.inference import annotate, annotate_xyxy
import numpy as np


import requests
from PIL import Image
import base64

import torch
import torchvision.transforms.functional as F

import pycocotools.mask as mask_util


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # import ipdb; ipdb.set_trace()
    # resize if needed
    w, h = img.size
    if max(h, w) > 800:
        if h > w:
            new_h = 800
            new_w = int(w * 800 / h)
        else:
            new_w = 800
            new_h = int(h * 800 / w)
        # import ipdb; ipdb.set_trace()
        img = F.resize(img, (new_h, new_w))
    return img

def encode(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def main():
    model_name = args.model_name

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return

    headers = {"User-Agent": "FastChat Client"}
    if args.send_image:
        img = load_image(args.image_path)
        img_arg = encode(img)
    else:
        img_arg = args.image_path
        img = None
    datas = {
        "model": model_name,
        "caption": args.caption,
        "image": img_arg,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }
    tic = time.time()
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=datas,
    )
    toc = time.time()
    print(f"Time: {toc - tic:.3f}s")

    print("detection result:")
    print(response.json())
    # response is 'Response' with :
    # ['_content', '_content_consumed', '_next', 'status_code', 'headers', 'raw', 'url', 'encoding', 'history', 'reason', 'cookies', 'elapsed', 'request', 'connection', '__module__', '__doc__', '__attrs__', '__init__', '__enter__', '__exit__', '__getstate__', '__setstate__', '__repr__', '__bool__', '__nonzero__', '__iter__', 'ok', 'is_redirect', 'is_permanent_redirect', 'next', 'apparent_encoding', 'iter_content', 'iter_lines', 'content', 'text', 'json', 'links', 'raise_for_status', 'close', '__dict__', '__weakref__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']

    # visualize
    res = response.json()
    # import ipdb; ipdb.set_trace()
    boxes = torch.Tensor(res["boxes"])
    logits =  torch.Tensor(res["logits"])
    phrases = res["phrases"]
    if img is not None:
        image_source = np.array(img.convert("RGB"))
    else:
        image_source = np.array(Image.open(args.image_path).convert("RGB"))
    annotated_frame = annotate_xyxy(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite("annotated_image.jpg", annotated_frame)

    # show mask
    masks_rle = res["masks_rle"]
    for mask_rle in masks_rle:
        mask = mask_util.decode(mask_rle)
        mask = torch.Tensor(mask)
        annotated_frame = show_mask(mask, annotated_frame)
    cv2.imwrite("annotated_image_mask.jpg", annotated_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='grounded_sam')

    # model parameters
    parser.add_argument(
        "--caption", type=str, default="dogs ."
    )
    parser.add_argument(
        "--image_path", type=str, default="/home/liushilong/code/GroundingFolder/Grounded-Segment-Anything/assets/demo2.jpg"
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3,
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25,
    )
    parser.add_argument(
        "--send_image", action="store_true",
    )
    args = parser.parse_args()

    main()
