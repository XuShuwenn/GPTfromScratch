import torch
from torch.nn import Module, Linear, LayerNorm, Dropout, MultiheadAttention, Embedding, ModuleList
from torch.nn.functional import gelu
from torch.nn.init import xavier_uniform_
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    learning_rate: float = 3e-4
    lr_decay_steps: int = 1000
    lr_decay_rate: float = 0.95
    logging_steps: int = 100

class GPT(Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.d_ff = config.d_ff
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout

        # 添加统计计数器
        self.token_clip_count = 0
        self.position_clip_count = 0
        self.batch_count = 0
        
        self.embedding = Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = Embedding(config.max_seq_len, config.d_model)
        self.layers = ModuleList([GPTLayer(config.d_model, config.n_heads, config.d_ff, config.dropout, config.max_seq_len) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embedding):
            xavier_uniform_(module.weight)

    def forward(self, x, attention_mask=None, output_attentions=False, output_activations=False):
        # 在这里直接使用类内部实例变量获取vocab_size，而不是通过self.config
        vocab_size = self.embedding.num_embeddings
        max_pos_len = self.pos_embedding.num_embeddings
        
        batch_size, seq_len = x.size()
        self.batch_count += 1
        
        # 验证输入token的有效性，并记录哪些需要裁剪
        token_clipped_in_batch = False
        if torch.any(x >= vocab_size):
            invalid_indices = torch.where(x >= vocab_size)
            invalid_values = x[invalid_indices]
            x = torch.clamp(x, 0, vocab_size - 1)
            if len(invalid_values) > 0:
                token_clipped_in_batch = True
                self.token_clip_count += 1
                if self.batch_count % 50 == 1 or self.token_clip_count <= 5:  # 减少日志输出频率
                    print(f"警告: 检测到无效的token ID: {invalid_values.tolist()[:5]}... (共{len(invalid_values)}个), 已被裁剪到有效范围")
        
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 验证位置索引的有效性
        position_clipped_in_batch = False
        if torch.any(pos >= max_pos_len):
            pos = torch.clamp(pos, 0, max_pos_len - 1)
            position_clipped_in_batch = True
            self.position_clip_count += 1
            if self.batch_count % 50 == 1 or self.position_clip_count <= 5:  # 减少日志输出频率
                print(f"警告: 位置索引超出范围，已被裁剪到 {max_pos_len - 1}")
                print(f"统计: 总批次数: {self.batch_count}, Token裁剪批次数: {self.token_clip_count}, 位置裁剪批次数: {self.position_clip_count}")
            
        x = self.embedding(x) + self.pos_embedding(pos)
        
        attentions = [] if output_attentions else None
        activations = [] if output_activations else None
        
        for i, layer in enumerate(self.layers):
            x, attn_weights, layer_activations = layer(x, attention_mask, output_attentions, output_activations)
            if output_attentions and attn_weights is not None:
                attentions.append(attn_weights)
            if output_activations and layer_activations is not None:
                activations.append(layer_activations)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        outputs = {'logits': logits}
        if output_attentions:
            outputs['attentions'] = attentions
        if output_activations:
            outputs['activations'] = activations
            
        return outputs

class GPTLayer(Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 d_ff, 
                 dropout=0.1,
                 max_seq_len=1024):
        super(GPTLayer, self).__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x, attention_mask=None, output_attentions=False, output_activations=False):
        # Causal mask
        batch_size, seq_len = x.size(0), x.size(1)
        device = x.device
        
        # Create a causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        
        # The mask should be of shape (batch_size, seq_len, seq_len) for MultiheadAttention
        # but it will be broadcasted correctly from (seq_len, seq_len)
        attn_mask = causal_mask

        x_ln = self.ln_1(x)
        attn_output, attn_weights = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=output_attentions, is_causal=False) # is_causal=False because we provide our own mask
        x = x + attn_output
        
        # MLP部分
        mlp_input = self.ln_2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output
        
        # 收集激活值
        activations = None
        if output_activations:
            activations = {
                'post_attention': x.detach().clone(),
                'mlp_hidden': self.mlp.get_hidden_states() if hasattr(self.mlp, 'get_hidden_states') else mlp_output.detach().clone()
            }
        
        return x, attn_weights if output_attentions else None, activations

class MLP(Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        self.hidden_states = None

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        self.hidden_states = x.detach().clone()
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_hidden_states(self):
        return self.hidden_states

