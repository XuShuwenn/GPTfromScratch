#!/bin/bash
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES="3"
TOKENIZERS_PARALLELISM=false

# 清理旧的实验数据文件
if [ -f "logs/GPT2-49M/metrics.csv" ]; then
    rm -rf logs/GPT2-49M/*
fi

# 改进的GPT2-49M训练脚本
accelerate launch --num_processes=1 train.py \
    --model_name="GPT2-49M" \
    --vocab_size=50257 \
    --d_model=384 \
    --n_layers=6 \
    --n_heads=6 \
    --d_ff=1536 \
    --max_seq_len=256 \
    --batch_size=96 \
    --gradient_accumulation_steps=1 \
    --epochs=5 \
    --learning_rate=1e-4 \
    --min_learning_rate=1e-6 \
    --weight_decay=0.05 \
    --dropout=0.2 \
    --grad_clip=0.5 \
    --logging_steps=1000 \
    --eval_steps=10000 \
    --train_file="data/tinystories/tokenized_train_bs256" \
    --val_file="data/tinystories/tokenized_validation_bs256" \
    --log_dir="logs"