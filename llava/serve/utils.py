from typing import Tuple, List

import re
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image


def annotate_xyxy(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = boxes.numpy()
    detections = sv.Detections(xyxy=xyxy)

    # labels = [
    #     f"{phrase} {logit:.2f}"
    #     for phrase, logit
    #     in zip(phrases, logits)
    # ]
    labels = []
    for i in range(len(boxes)):
        anno = ''
        if phrases is not None:
            anno += phrases[i]
        if logits is not None:
            if len(anno) > 0:
                anno += ' '
            anno += f'{logits[i]:.2f}'
        labels.append(anno)

    box_annotator = sv.BoxAnnotator()
    # annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = image_source
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def show_mask(mask: torch.Tensor, image: Image, random_color=True) -> Image:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = image.convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    # resize
    img_w, img_h = annotated_frame_pil.size
    mask_image_pil = mask_image_pil.resize((img_w, img_h), Image.BILINEAR)

    return Image.alpha_composite(annotated_frame_pil, mask_image_pil).convert("RGB")
