
#!/bin/bash
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES="3"
TOKENIZERS_PARALLELISM=false

# 为accelerate指定正确的GPU数量
accelerate launch --num_processes=1 train.py \
    --model_name="GPT2-162M" \
    --vocab_size=50257 \
    --d_model=768 \
    --n_layers=12 \
    --n_heads=12 \
    --d_ff=3072 \
    --max_seq_len=256 \
    --batch_size=64 \
    --gradient_accumulation_steps=1 \
    --epochs=10 \
    --learning_rate=2e-3 \
    --min_learning_rate=6e-5 \
    --weight_decay=0.01 \
    --dropout=0.1 \
    --grad_clip=1.0 \
    --logging_steps=50 \
    --eval_steps=500 \
    --train_file="data/tinystories/train" \
    --val_file="data/tinystories/validation" \
    --log_dir="logs"