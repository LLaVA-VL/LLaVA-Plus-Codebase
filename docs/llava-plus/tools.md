# LLaVA-Plus Server

![llava-plus-arch](../../images/llava-plus-arch.png)

As shown in the figure above, we need to build an API cloud to main all tools. 

## Launch a controller
Build a controller like the guides in [readme](README.md#1-Launch-a-controller).

Our tool workers and llm-model workers share the same controller.

## Launch tool workers: An Example
We provide all model workers in the `serve/` folder. The tool worker is named as `{tool_name}_worker.py`. To run a worker, make sure you have installed the required tool(or copy the worker file to different tool folders).

An example for the Grounded-SAM, whch includes Grounding-DINO, SAM, and Grounded-SAM.

```sh
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
python -m pip install -e GroundingDINO
python -m pip install -e segment_anything
python serve/grounding_dino_worker.py
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". You need to open another terminal process for other operations.

Test the worker:
```sh
python serve/grounding_dino_test_message.py
```

## All Tools

| Tool Name       | Install Source                                                                                                                                     |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Grounding DINO  | [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)                           |
| SAM             | [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)                           |
| Grounded-SAM    | [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)                           |
| CLIP-Retrieval  | [https://github.com/rom1504/clip-retrieval](https://github.com/rom1504/clip-retrieval)                                                             |
| InstructPix2Pix | [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)                                                               |
| StableDiffusion | [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)                                                               |
| BLIP2           | [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)                                                         |
| RAM             | [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)                           |
| Semantic-SAM    | [https://github.com/UX-Decoder/Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)                                                           |
| SEEM            | [https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) |
| OCR(easyocr)    | [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)                                                                         |