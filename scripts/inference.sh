#!/bin/bash

# 推理参数
model_name="GPT2-8M"
# 请将此路径替换为您训练好的最佳模型检查点
model_path="logs/GPT2-8M/ckpts/model_epoch9_valloss2.5000.pth" 
prompt="Once upon a time, in a land far, far away"
max_length=128
temperature=0.8
top_k=25


# 运行推理脚本
python inference.py \
    --model_name "$model_name" \
    --model_path "$model_path" \
    --prompt "$prompt" \
    --max_length "$max_length" \
    --temperature "$temperature" \
    --top_k "$top_k"