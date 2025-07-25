#!/bin/bash
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES="2"
TOKENIZERS_PARALLELISM=false

# 清理旧的实验数据文件
if [ -f "logs/GPT2-29M/metrics.csv" ]; then
    rm -rf logs/GPT2-29M/*
fi

# 使用更保守的超参数来避免过拟合

accelerate launch --num_processes=1 train.py \
    --model_name="GPT2-29M" \
    --vocab_size=50257 \
    --d_model=256 \
    --n_layers=4 \
    --n_heads=4 \
    --d_ff=1024 \
    --max_seq_len=256 \
    --batch_size=128 \
    --gradient_accumulation_steps=1 \
    --epochs=5 \
    --learning_rate=1e-4 \
    --min_learning_rate=1e-6 \
    --weight_decay=0.03 \
    --dropout=0.1 \
    --grad_clip=0.5 \
    --logging_steps=250 \
    --eval_steps=5000 \
    --train_file="data/tinystories/train" \
    --val_file="data/tinystories/validation" \
    --log_dir="logs"