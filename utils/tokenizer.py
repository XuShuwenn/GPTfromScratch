import torch

class GPT2Tokenizer:
    """
    A simple tokenizer that tokenizes text into characters.
    """
    def __init__(self, vocab_file=None):
        # 简单实现：基于字符级别的tokenizer
        # vocab_file: 可选，若有则加载词表，否则自动构建
        self.vocab = {}
        self.inv_vocab = {}
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            # 默认构建基础字符词表
            self.vocab = {chr(i): i for i in range(32, 127)}
            self.vocab['<pad>'] = 0
            self.vocab['<unk>'] = 1
            self.vocab['<bos>'] = 2
            self.vocab['<eos>'] = 3
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def load_vocab(self, vocab_file):
        self.vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=True, max_length=None, padding=False, truncation=False):
        # 字符级别编码
        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab.get('<bos>', 2))
        for ch in text:
            tokens.append(self.vocab.get(ch, self.vocab.get('<unk>', 1)))
        if add_special_tokens:
            tokens.append(self.vocab.get('<eos>', 3))
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        if padding and max_length is not None:
            while len(tokens) < max_length:
                tokens.append(self.vocab.get('<pad>', 0))
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for idx in token_ids:
            token = self.inv_vocab.get(idx, '<unk>')
            if skip_special_tokens and token in ['<pad>', '<unk>', '<bos>', '<eos>']:
                continue
            tokens.append(token)
        return ''.join(tokens)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False):
        # 用于兼容HuggingFace风格
        input_ids = self.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding=padding,
            truncation=truncation
        )
        if return_tensors == 'pt':
            import torch
            input_ids = torch.tensor([input_ids], dtype=torch.long)
            return {'input_ids': input_ids}
        return {'input_ids': input_ids}
