# example usage: bash playground/llava-plus-data/grounding/run.sh
# for grounding data generations
python playground/data_tools_v3/grounding/generate.py generate_data \
    --output_file="data/LLM_data/llava_plus_v3/grounding.jsonl" \
    --sample_json="playground/llava-plus-data/grounding/question_example_grounding.json" \
    --seed=3551 \
    --num_examples=5000 \
    --num_workers=32 

# for data generations
python playground/data_tools_v3/grounding/generate.py generate_data \
    --output_file="data/LLM_data/llava_plus_v3/grounding_description.jsonl" \
    --sample_json="playground/llava-plus-data/grounding/question_example_grounding_description.json" \
    --seed=32132 \
    --num_examples=5000 \
    --num_workers=32