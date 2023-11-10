export LLAVA_DEFAULT_CONVERSATION="conv_vicuna_v1"

# run
export LOGDIR=logs/
export out_dir="${LOGDIR}/llava-plus/llava_plus_v1.3_7b"
mkdir -p $out_dir
echo ${out_dir}/loginfo.txt

# Note: Our scripts support multi-source data and image folders. Seperate each item with `,`. Note that it may cause problems if multiple folders have images with the same name.

deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path path/to/LLAVA/vicuna-7b-v1.3 \
    --pretrain_mm_mlp_adapter path/to/llava-pretrain-vicuna-7b-v1.3  \
    --version v1 \
    --data_path path/to/llava-150k-tool-aug.json,path/to/llava-plus-v1-117k-tool-merge.json \
    --image_folder /path/to/coco/train2017/,/path/to/hiertext/train,/path/to/infoseek/infoseek_images,/path/to/instruct-pix2pix/clip-filtered-dataset,/path/to/goldg/vg_mdetr/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    >> ${out_dir}/loginfo.txt 2>&1