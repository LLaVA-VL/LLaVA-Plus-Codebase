import json
from typing import Dict, List
from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255)
                                  for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] -
                     self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(
            output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(
                output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def reorganize_source_for_tool_use(source: List[Dict]):
    """
    merge thoughts, actions, value into value, and add prefixs to value.

    Args:
        source (List[Dict]): A list of dict, each dict is a conversation, has keys: from, value, Optional[thoughts, actions, ...]

    Returns:
        List[Dict]: A list of dict, each dict is a conversation, has keys: from, value, ...
            value is a string with thougths and value merged, with prefixs.
    """
    new_source = []
    for conv in source:
        if conv['from'].lower() == 'human':
            new_source.append(conv)
            continue
        mid_sentence = ""
        if "thoughts" in conv:
            mid_sentence = mid_sentence + \
                "\"{}\" {}".format("thoughts🤔", conv["thoughts"]) + "\n"
            conv.pop("thoughts")
        if "actions" in conv:
            mid_sentence = mid_sentence + \
                "\"{}\" {}".format(
                    "actions🚀", json.dumps(conv["actions"])) + "\n"
            conv.pop("actions")
        if "value" in conv:
            mid_sentence = mid_sentence + \
                "\"{}\" {}".format("value👉", conv["value"]) + "\n"
            conv.pop("value")
        conv['value'] = mid_sentence
        new_source.append(conv)
    return new_source


def reorganize_source_for_tool_use_batch(sources: List[List[Dict]]):
    # batch version of reorganize_source_for_tool_use
    return [reorganize_source_for_tool_use(source) for source in sources]
