
def calculate_gpt_params(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
    # Embedding层（无bias）
    embedding_params = vocab_size * d_model
    pos_embedding_params = max_seq_len * d_model

    # Transformer Block（每层）
    # 1. 两个LayerNorm（weight+bias）
    layernorm_params = 2 * (2 * d_model)
    # 2. MultiheadAttention
    #   - qkv: 3 * d_model * d_model
    #   - out: d_model * d_model
    #   - bias: 4 * d_model
    attn_params = 3 * d_model * d_model + d_model * d_model + 4 * d_model
    # 3. MLP
    #   - fc1: d_model * d_ff + d_ff
    #   - fc2: d_ff * d_model + d_model
    mlp_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
    # 每层总参数
    block_params = layernorm_params + attn_params + mlp_params
    all_blocks_params = n_layers * block_params

    # 输出层
    ln_f_params = 2 * d_model
    lm_head_params = d_model * vocab_size + vocab_size

    total_params = (
        embedding_params +
        pos_embedding_params +
        all_blocks_params +
        ln_f_params +
        lm_head_params
    )
    
    return int(total_params)