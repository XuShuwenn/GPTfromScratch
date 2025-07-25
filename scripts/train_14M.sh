#!/bin/bash
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES="4"
TOKENIZERS_PARALLELISM=false

# 清理旧的实验数据文件
if [ -f "logs/GPT2-14M/metrics.csv" ]; then
    rm -rf logs/GPT2-14M/*
fi

# 为accelerate指定正确的GPU数量
accelerate launch --num_processes=1 train.py \
    --model_name="GPT2-14M" \
    --vocab_size=50257 \
    --d_model=128 \
    --n_layers=4 \
    --n_heads=4 \
    --d_ff=512 \
    --max_seq_len=256 \
    --batch_size=128 \
    --gradient_accumulation_steps=1 \
    --epochs=5 \
    --learning_rate=1e-4 \
    --min_learning_rate=1e-6 \
    --weight_decay=0.01 \
    --dropout=0.1 \
    --grad_clip=0.5 \
    --logging_steps=125 \
    --eval_steps=2500 \
    --train_file="data/tinystories/tokenized_train_bs256" \
    --val_file="data/tinystories/tokenized_validation_bs256" \
    --log_dir="logs"