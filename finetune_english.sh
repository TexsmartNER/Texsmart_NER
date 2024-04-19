export NUM_NODE=1
export NUM_TRAINERS=1

model_path="BASE_MODEL_PATH"
data_path="ner/en"
output_dir="checkpoint_ft_en"
lang="en"

torchrun --nnodes=$NUM_NODE --nproc_per_node=$NUM_TRAINERS \
    finetune.py \
    --model_name_or_path ${model_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_max_length 1536 \
    --language ${lang} \
    --from_scratch False \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --warmup_ratio 0.03 \
    --optim "adamw_torch" \
    --fp16 False \
    --seed 1234 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 6 \
    --save_total_limit 5 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --remove_unused_columns False \
    --report_to "tensorboard"
