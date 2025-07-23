
#!/bin/bash

python -m train.py \
    --vocab_size=50257 \
    --d_model=384 \
    --n_layers=6 \
    --n_heads=6 \
    --d_ff=1536 \
    --max_seq_len=1024 \
    --batch_size=64 \
    --gradient_accumulation_steps=2 \
    --epochs=5 \
    --learning_rate=1e-3 \
    --min_learning_rate=1e-4 \
    --lr_decay_steps=2000 \
    --lr_decay_rate=0.9 \
    --warmup_steps=1000 \
    --weight_decay=0.1 \
    --dropout=0.1 \
    --logging_steps=50 \
    --eval_steps=500 \
    --train_file=hf:train \
    --val_file=hf:validation \
    --model_save_path=output/model_8M.pth \
    --save_steps=1000 \
    --save_weights_steps=1000 \
    --save_model_steps=1000 \
    --grad_clip=1.0 \