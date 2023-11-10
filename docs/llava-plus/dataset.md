# LLaVA-Plus Data

## Released Data

[huggingface](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data)

- [llava-150k-tool-aug.json](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data/blob/main/llava-150k-tool-aug.json) augment the llava-insttrution-150 with extrac `"thoughts"` and `"actions"` to ensure the data format as llava-plus required.
- [llava-plus-v1-117k-tool-merge.json](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data/blob/main/llava-plus-v1-117k-tool-merge.json) is tool learning visual instruction data by prompting ChatGPT/GPT-4.


## How to build the Instruction Data
We provide an example to constuct grounding data [here](playground/llava-plus-data/grounding/run.sh).