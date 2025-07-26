#!/bin/bash

# GPT 模型推理脚本
# 支持三种模型: GPT2-14M, GPT2-29M, GPT2-49M

set -e  # 遇到错误时退出

# 默认参数
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0
MODEL_NAME="GPT2-49M"
PROMPT="Once upon a time, there was a little girl who loved to play in the garden."
MAX_LENGTH=100
TEMPERATURE=0.7
TOP_K=20
USE_BEST_CKPT=true

# 显示使用帮助
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_NAME     Model to use (GPT2-14M, GPT2-29M, GPT2-49M) [default: GPT2-14M]"
    echo "  -p, --prompt PROMPT        Input prompt for generation [default: 'Once upon a time']"
    echo "  -l, --max-length LENGTH    Maximum generation length [default: 100]"
    echo "  -t, --temperature TEMP     Sampling temperature [default: 0.7]"
    echo "  -k, --top-k TOP_K         Top-k sampling parameter [default: 20]"
    echo "  --model-path PATH         Specific checkpoint path (overrides auto-selection)"
    echo "  --list-ckpts              List available checkpoints for a model"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m GPT2-49M -p \"Hello world\" -l 50"
    echo "  $0 --model GPT2-14M --prompt \"In a distant galaxy\" --temperature 0.8"
    echo "  $0 --list-ckpts -m GPT2-29M"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -l|--max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            USE_BEST_CKPT=false
            shift 2
            ;;
        --list-ckpts)
            LIST_CKPTS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证模型名称
case $MODEL_NAME in
    GPT2-14M|GPT2-29M|GPT2-49M)
        ;;
    *)
        echo "Error: Invalid model name '$MODEL_NAME'"
        echo "Supported models: GPT2-14M, GPT2-29M, GPT2-49M"
        exit 1
        ;;
esac

# 设置模型配置参数
case $MODEL_NAME in
    GPT2-14M)
        VOCAB_SIZE=50257
        D_MODEL=128
        N_LAYERS=4
        N_HEADS=4
        D_FF=512
        MAX_SEQ_LEN=256
        ;;
    GPT2-29M)
        VOCAB_SIZE=50257
        D_MODEL=256
        N_LAYERS=4
        N_HEADS=4
        D_FF=1024
        MAX_SEQ_LEN=256
        ;;
    GPT2-49M)
        VOCAB_SIZE=50257
        D_MODEL=384
        N_LAYERS=6
        N_HEADS=6
        D_FF=1536
        MAX_SEQ_LEN=256
        ;;
esac

CKPT_DIR="logs/$MODEL_NAME/ckpts"

# 列出检查点
if [[ "$LIST_CKPTS" == "true" ]]; then
    echo "Available checkpoints for $MODEL_NAME:"
    if [[ -d "$CKPT_DIR" ]]; then
        ls -la "$CKPT_DIR"/*.pth 2>/dev/null || echo "No checkpoints found in $CKPT_DIR"
    else
        echo "Checkpoint directory $CKPT_DIR does not exist"
    fi
    exit 0
fi

# 自动选择最佳检查点
if [[ "$USE_BEST_CKPT" == "true" ]]; then
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "Error: Checkpoint directory $CKPT_DIR does not exist"
        echo "Please train the model first or specify a manual checkpoint path with --model-path"
        exit 1
    fi
    
    # 找到验证损失最低的检查点
    MODEL_PATH=$(ls "$CKPT_DIR"/*.pth 2>/dev/null | head -1)
    if [[ -z "$MODEL_PATH" ]]; then
        echo "Error: No checkpoints found in $CKPT_DIR"
        echo "Please train the model first or specify a manual checkpoint path with --model-path"
        exit 1
    fi
    
    # 选择验证损失最低的检查点（简化版本，使用字符串排序）
    BEST_CKPT=""
    BEST_LOSS="999999"
    for ckpt in "$CKPT_DIR"/*.pth; do
        if [[ -f "$ckpt" ]]; then
            # 从文件名中提取验证损失
            loss=$(basename "$ckpt" | grep -o 'valloss[0-9]*\.[0-9]*' | sed 's/valloss//')
            # 使用awk进行浮点数比较
            if [[ -n "$loss" ]] && awk "BEGIN {exit !($loss < $BEST_LOSS)}"; then
                BEST_LOSS=$loss
                BEST_CKPT="$ckpt"
            fi
        fi
    done
    
    if [[ -z "$BEST_CKPT" ]]; then
        # 如果没找到最佳的，就用第一个
        BEST_CKPT=$(ls "$CKPT_DIR"/*.pth 2>/dev/null | head -1)
    fi
    
    MODEL_PATH="$BEST_CKPT"
fi

# 验证检查点文件存在
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Checkpoint file $MODEL_PATH does not exist"
    exit 1
fi

# 显示配置信息
echo "================================"
echo "GPT Model Inference Configuration"
echo "================================"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $MODEL_PATH"
echo "Prompt: \"$PROMPT\""
echo "Max Length: $MAX_LENGTH"
echo "Temperature: $TEMPERATURE"
echo "Top-K: $TOP_K"
echo "Model Config:"
echo "  - Vocab Size: $VOCAB_SIZE"
echo "  - D Model: $D_MODEL"
echo "  - N Layers: $N_LAYERS"
echo "  - N Heads: $N_HEADS"
echo "  - D FF: $D_FF"
echo "  - Max Seq Len: $MAX_SEQ_LEN"
echo "================================"

# 运行推理
python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --vocab_size $VOCAB_SIZE \
    --d_model $D_MODEL \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --d_ff $D_FF \
    --max_seq_len $MAX_SEQ_LEN \
    --dropout 0.1

echo ""
echo "Inference completed!"
