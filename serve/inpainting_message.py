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

def get_worker_addr(model_name, args):
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        _ = ret.json()
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    return worker_addr
def main():
    model_name = args.model_name


    sam_worker_addr = get_worker_addr("sam", args)
    inpainting_worker_addr = get_worker_addr("inpainting", args)
    if sam_worker_addr == "" or inpainting_worker_addr == "":
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
        "image": img_arg,
        "boxes": args.boxes,
    }
    tic = time.time()
    response = requests.post(
        sam_worker_addr + "/worker_generate",
        headers=headers,
        json=datas,
    )
    toc = time.time()
    print(f"Time: {toc - tic:.3f}s")

    print("segmentation result:")
    print(response.json())

    # visualize
    res = response.json()
    res['boxes'] = args.boxes
    boxes = torch.Tensor(res["boxes"])
    logits =  torch.ones(boxes.shape[0])
    phrases = ['' for _ in range(boxes.shape[0])]
    if img is not None:
        image_source = np.array(img.convert("RGB"))
    else:
        image_source = np.array(Image.open(args.image_path).convert("RGB"))
    annotated_frame = annotate_xyxy(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite("annotated_image.jpg", annotated_frame)

    # show mask
    masks_rle_list = res["masks_rle"]
    for mask_rle in masks_rle_list:
        mask = mask_util.decode(mask_rle)
        mask = torch.Tensor(mask)
        annotated_frame = show_mask(mask, annotated_frame)
    cv2.imwrite("inpainting_1.jpg", annotated_frame)

    _mask = Image.fromarray(mask_util.decode(masks_rle_list[1]) * 255)
    _mask.save("inpainting_mask.jpg")
    # import ipdb; ipdb.set_trace()

    # inpainting
    datas = {
        "image": img_arg,
        "mask": masks_rle_list[1],
        "prompt": args.prompt,
    }

    tic = time.time()
    response = requests.post(
        inpainting_worker_addr + "/worker_generate",
        headers=headers,
        json=datas,
    )
    toc = time.time()
    print(f"Time: {toc - tic:.3f}s")

    res = response.json()
    image = Image.open(BytesIO(base64.b64decode(res['edited_image']))).convert("RGB")
    # import ipdb; ipdb.set_trace()
    # save
    image.save("inpainting_2.jpg")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='sam')

    # model parameters
    parser.add_argument(
        "--prompt", type=str, default="flowers"
    )
    parser.add_argument(
        "--boxes",
        default="[[0.1, 0.2, 0.8, 0.5], [0.3, 0.5, 0.9, 0.9]]",
        type=lambda s: eval(s),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--image_path", type=str, default="assets/demo2.jpg"
    )
    parser.add_argument(
        "--send_image", action="store_true",
    )
    args = parser.parse_args()

    main()
