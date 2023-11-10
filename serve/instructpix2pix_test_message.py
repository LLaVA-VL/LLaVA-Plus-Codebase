"""Send a test message."""
import argparse
import json
import time
from io import BytesIO
import cv2
from groundingdino.util.inference import annotate
import numpy as np


import requests
from PIL import Image
import base64

import torch
import torchvision.transforms.functional as F

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

    headers = {"User-Agent": "GSAM Client"}
    if args.send_image:
        img = load_image(args.image_path)
        img_arg = encode(img)
    else:
        img_arg = args.image_path
    datas = {
        "model": model_name,
        "image": img_arg,
        "prompt": "Hi AI! make them like a cat"
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
    res = response.json()

    image = Image.open(BytesIO(base64.b64decode(res['edited_image']))).convert("RGB")
    # save
    image.save("demo2_edited2.jpg")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='ip2p')

    # model parameters
    parser.add_argument(
        "--image_path", type=str, default="78851692243550_.pic.jpg"
    )
    parser.add_argument(
        "--send_image", action="store_true",
    )
    args = parser.parse_args()

    main()
