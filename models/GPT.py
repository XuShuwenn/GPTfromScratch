import torch
from torch.nn import Module, Linear, LayerNorm, Dropout, MultiheadAttention
from torch.nn.functional import softmax, linear
from torch.nn.init import xavier_uniform_

class GPT(Module):
    def __init__(self,
                 vocab_size, 
                 d_model, 
                 n_layers, 
                 n_heads, 
                 d_ff, 
                 max_seq_len, 
                 dropout=0.1):
        super(GPT, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = Embedding(max_seq_len, d_model)
        self.layers = ModuleList([GPTLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (Linear, LayerNorm)):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

class GPTLayer(Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 d_ff, 
                 dropout=0.1):
        super(GPTLayer, self).__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class MLP(Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Embedding(Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = Linear(vocab_size, d_model)
        self.pos_embedding = Linear(max_seq_len, d_model)

    def forward(self, x):
        return self.embedding(x) + self.pos_embedding(x)

