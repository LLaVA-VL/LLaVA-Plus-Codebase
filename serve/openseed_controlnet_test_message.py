"""Send a test message."""
import argparse
import time
from io import BytesIO
import io
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

    headers = {"User-Agent": "FastChat Client"}
    img_arg = args.image_path
    image = load_image(img_arg).convert('RGB')
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')
    image_data_bytes = image_data.getvalue()
    # 将图像数据编码为Base64字符串
    encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
    mode=args.mode
    datas = {
        "caption": args.caption,
        "image": encoded_image,
        "mask":None,
        "mode":mode
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
    image_seg = res["image_seg"]
    image_result =  res["image"]
    if image_seg is not None:
        image_seg=Image.open(io.BytesIO(base64.b64decode(image_seg))).convert("RGB")
        image_seg.save("output/seg.jpg")
    if image_seg is not None:
        image_result=Image.open(io.BytesIO(base64.b64decode(image_result))).convert("RGB")
        image_result.save("output/result.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str, default="http://localhost:22003")
    parser.add_argument("--model-name", type=str, default='grounding_dino')
    parser.add_argument("--mode", type=str, default='openseed')
    # parser.add_argument("--model-name", type=str, default='grounding_dino')

    # model parameters
    parser.add_argument(
        "--caption", type=str, default="house"
    )
    parser.add_argument(
        "--image_path", type=str, default="house.png"
    )
    args = parser.parse_args()

    main()
