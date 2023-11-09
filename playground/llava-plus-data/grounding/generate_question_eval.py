import os
import json
import time
from tqdm import tqdm
import fire

import openai

import concurrent.futures
import random
import json
import time
from collections import Counter
from functools import partial

from pycocotools.coco import COCO

import requests
from PIL import Image
import base64
import json
import time
from io import BytesIO

import torchvision.transforms.functional as F


# vars
controller_address = "http://localhost:21001"
model_name = 'grounding_dino'

def get_openai_api():
    api_type = os.environ.get('API_TYPE', 'openai')
    if api_type == 'azure':
        api_key = os.environ.get('API_KEY', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        engine = os.environ.get('ENGINE', 'gpt-35-turbo')
        api_base = os.environ.get('API_BASE')

        return {
            'api_type': 'azure',
            'api_version': '2023-03-15-preview',
            'engine': engine,
            'api_key': api_key,
            'api_base': f'https://{api_base}.openai.azure.com',
        }
    else:
        api_key = os.environ.get('API_KEY', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = os.environ.get('MODEL', 'gpt-4')

        return {
            'model': model,
            'api_key': api_key,
        }


def ask_gpt(messages, max_retries=35, temperature=0.2, top_p=0.9, max_tokens=512):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    openai_kwargs = get_openai_api()

    for i in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                **openai_kwargs,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            if os.getenv('DEBUG_PRINT'):
                print(response['choices'][0]['message']['content'])
            return response['choices'][0]['message']['content']
        except Exception as e:
            if type(e) in [openai.error.InvalidRequestError, KeyError]:
                print(type(e), e)
                return None
            print(type(e), e)
            time.sleep(2)
            continue


def R(x):
    if isinstance(x, list):
        return [R(i) for i in x]
    elif isinstance(x, dict):
        return {k: R(v) for k, v in x.items()}
    elif isinstance(x, float):
        return round(x, 2)
    
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
    buffered.close()
    return img_b64_str

def get_worker_addr(controller_addr, model_name):
    # get worker_addr
    # ret = requests.post(controller_addr + "/refresh_all_workers")
    # ret = requests.post(controller_addr + "/list_models")
    # models = ret.json()["models"]
    # models.sort()
    # print(f"Models: {models}")

    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    del ret
    # print(f"worker_addr: {worker_addr}")
    return worker_addr

def generate_worker(captions_strs, objects_strs, examples, sample, image_dir):
    # 1. captions_strs + objects_strs -> questions
    # 2. questions -> grounding dino input
    # 3. grounding dino input -> grounding dino output
    # 4. captions_strs + objects_strs + questions + grounding dino output -> answer

    # 1. captions_strs + objects_strs -> questions
    messages = [
        {'role': 'system', 'content': """You are an AI visual assistant that can analyze a single image. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Generate a question that users may be interested to ask about the image. The question should ask the AI to detect some objects in the image. The question should be answerable by the given sentences and the given object locations.
The question should ask the AI to detect some objects in the image."""},
    {"role": "user", "content": examples[0]['captions']+'\n'+examples[0]['objects']},
    {"role": "assistant", "content": examples[0]['question']},
    {"role": "user", "content": examples[1]['captions']+'\n'+examples[1]['objects']},
    {"role": "assistant", "content": examples[1]['question']},
    {"role": "user", "content": captions_strs + '\n' + objects_strs}
    ]
    question = ask_gpt(messages, temperature=0.9, top_p=0.95)
    if question is None:
        print("question is None, return None")
        return None

    # return
    return {
        "unique_id": str(time.time()) + '_' + str(sample['id']),
        "image_id": sample['id'],
        "image_file_name": sample['file_name'],
        "image_path": os.path.join(image_dir, sample['file_name']),
        "question": question,
    }


def generate_data(
    output_file,
    sample_json,
    overwrite=False,
    num_workers=1,
    num_examples=1000,
    coco_caption_path="/comp_robot/liushilong/data/coco/annotations/captions_{split}2014.json",
    coco_object_path="/comp_robot/liushilong/data/coco/annotations/instances_{split}2014.json",
    image_dir="/comp_robot/liushilong/data/coco/{split}2014",
    split='train',
    seed=23123,
    debug=False,
):
    # load existing data
    if not overwrite and os.path.exists(output_file):
        print("Loading existing data...")
        with open(output_file) as f:
            existing_examples = [json.loads(line) for line in f]
        print("Existing data loaded.")
        if len(existing_examples) >= num_examples:
            print("Enough examples, skip generating.")
            return
        print("Generating {} examples...".format(num_examples - len(existing_examples)))
        num_examples = num_examples - len(existing_examples)
        seed = seed + len(existing_examples)


    # load coco annos
    coco_cap = COCO(coco_caption_path.format(split=split))
    coco_obj = COCO(coco_object_path.format(split=split))
    image_dir = image_dir.format(split=split)

    # load coco images
    coco_images = coco_cap.loadImgs(coco_cap.getImgIds())
    coco_categories = coco_obj.loadCats(coco_obj.getCatIds())

    # random select 1000 images
    random.seed(seed)
    random.shuffle(coco_images)
    coco_images = coco_images[:num_examples]

    # load sample json
    with open(sample_json) as f:
        examples = json.load(f)

    # generate data
    print("Start generating data...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for sample_idx, sample in enumerate(coco_images):
            # load samples
            captions = coco_cap.loadAnns(coco_cap.getAnnIds(sample['id']))
            objects = coco_obj.loadAnns(coco_obj.getAnnIds(sample['id']))
            width, height = sample['width'], sample['height']
            for obj in objects:
                obj['bbox'] = [obj['bbox'][0] / width, obj['bbox'][1] / height, obj['bbox'][2] / width, obj['bbox'][3] / height]
                # xywh -> xyxy
                obj['bbox'][2] += obj['bbox'][0]
                obj['bbox'][3] += obj['bbox'][1]
                obj['bbox'] = R(obj['bbox'])

            captions_strs = "\n".join([cap['caption'].strip() for cap in captions])
            objects_strs = "\n".join([coco_obj.loadCats(obj['category_id'])[0]['name'] + ": " + str(obj['bbox']) for obj in objects])

            if debug:
                generate_worker(captions_strs, objects_strs, examples, sample, image_dir)
                continue


            futures[executor.submit(generate_worker, captions_strs, objects_strs, examples, sample, image_dir)] = sample_idx

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = open(output_file, 'a')
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result is None:
                time.sleep(0.1)
                continue
            writer.write(json.dumps(result) + '\n')
            writer.flush()
        writer.close()


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)