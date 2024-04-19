export NUM_NODE=1
export NUM_TRAINERS=8

tokenizer_path="bert-en-zh"
bert_path="bert-large-uncased"
data_path="PRETRAIN_DATA_PATH_STAGE1"
output_dir="checkpoints_stage1"

torchrun --nnodes=$NUM_NODE --nproc_per_node=$NUM_TRAINERS \
    pretrain.py \
    --model_name_or_path ${bert_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_max_length 128 \
    --stage 1 \
    --learning_rate 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --warmup_steps 0 \
    --warmup_steps 10000 \
    --max_steps 400000 \
    --optim "adamw_torch" \
    --fp16 True \
    --seed 1234 \
    --lr_scheduler_type "linear" \
    --per_device_train_batch_size 88 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --remove_unused_columns False \
    --report_to "tensorboard"

data_path="PRETRAIN_DATA_PATH_STAGE2"
bert_path="checkpoints_stage1/checkpoint-400000"
output_dir="checkpoints_stage2"

torchrun --nnodes=$NUM_NODE --nproc_per_node=$NUM_TRAINERS \
    pretrain.py \
    --model_name_or_path ${bert_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_max_length 512 \
    --stage 2 \
    --learning_rate 7e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_steps 80000 \
    --optim "adamw_torch" \
    --fp16 True \
    --seed 1234 \
    --lr_scheduler_type "linear" \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --remove_unused_columns False \
    --report_to "tensorboard"

data_path="PRETRAIN_DATA_PATH_STAGE3"
bert_path="checkpoints_stage2/checkpoint-80000"
output_dir="checkpoints_stage3"

torchrun --nnodes=$NUM_NODE --nproc_per_node=$NUM_TRAINERS \
    pretrain.py \
    --model_name_or_path ${bert_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --model_max_length 1536 \
    --stage 3 \
    --learning_rate 5e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_steps 20000 \
    --optim "adamw_torch" \
    --fp16 True \
    --seed 1234 \
    --lr_scheduler_type "linear" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --remove_unused_columns False \
    --report_to "tensorboard"