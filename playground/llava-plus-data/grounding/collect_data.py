import os
import json
import time
from tqdm import tqdm
import fire, random

tool_name = "grounding_dino"

def read_text_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return lines

def mention_tool(item):
    if 'grounding dino' in item['answer'].lower():
        return True
    
def add_image_token(text):
    return random.choice([
        '<image>\n' + text,
        text + '\n<image>',
    ])
    

def process_item(
    item,
    use_tools,
    thoughts,
):
    gd_output = item['grounding_dino_output']
    if 'size' in gd_output:
        gd_output.pop('size')
        
    if "<image>" not in item['question']:
        item['question'] = add_image_token(item['question'])
    new_item = {
        "unique_id": item['unique_id'],
        "image_id": item['image_id'],
        "file_name": item['image_file_name'],
        "data_source": "coco",
        "conversations": [
            {
                "from": "human",
                "value": item['question']
            },
            {
                "from": "gpt",
                "thoughts": thoughts,
                "actions": [{
                    "API_name": tool_name,
                    "API_params": {
                        "caption": item["grounding_dino_input"]
                    }
                }],
                "value": "I will use {tool_name} to help to answer the question. Please wait for a moment.".format(tool_name=tool_name)
            },
            {
                "from": "human",
                "value": f"{tool_name} output: {str(gd_output)}\n\nAnswer my first question: {item['question']}"
            },
            {
                "from": "gpt",
                "thoughts": f"Thanks to the output of {tool_name}. I can answer the question better.",
                "actions": [],
                "value": item['answer']
            }
        ]
    }
    return new_item

def collect_data(
    input_jsonl,
    save_path,
    use_tools=True,
    thought_examples_file="/home/liushilong/code/LM_many/LLaVA/playground/data_tools_additional/grounding/thoughts_examples.txt",
):
    res = []

    thought_list = read_text_file(thought_examples_file)

    # load jsonl
    num_filtered = 0
    with open(input_jsonl, "r") as f:
        for line in tqdm(f):
            line = json.loads(line)
            if mention_tool(line):
                num_filtered += 1
                continue
            new_item = process_item(
                line,
                use_tools,
                random.choice(thought_list),
            )
            res.append(new_item)

    # save
    with open(save_path, "w") as f:
        json.dump(res, f, indent=2)
    
    # print
    print("Save to {}".format(save_path))
    print("Number of filtered items: {}".format(num_filtered))
    print("Total number of items: {}".format(len(res)))
    print("Example:")
    print(res[0])

    

def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)