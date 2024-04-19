export NUM_NODE=1
export NUM_TRAINERS=1

# english evaluation
model_path="checkpoints/ckpt_en"
data_path="ner/en"
output_dir="eval_english"

torchrun --nnodes=$NUM_NODE --nproc_per_node=$NUM_TRAINERS \
    evaluate.py \
    --model_name_or_path ${model_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --language "en" \
    --per_device_eval_batch_size 16 \
    --remove_unused_columns False

# chinese evaluation 
model_path="checkpoints/ckpt_zh"
data_path="ner/zh"
output_dir="eval_chinese"

torchrun --nnodes=$NUM_NODE --nproc_per_node=$NUM_TRAINERS \
    evaluate.py \
    --model_name_or_path ${model_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --language "zh" \
    --per_device_eval_batch_size 16 \
    --remove_unused_columns False
