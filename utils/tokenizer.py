import torch
from transformers import GPT2Tokenizer as HFTokenizer

class GPT2Tokenizer:
    """
    A wrapper around HuggingFace GPT2Tokenizer for compatibility.
    """
    def __init__(self, vocab_file=None):
        # 使用HuggingFace的GPT-2 tokenizer
        self.tokenizer = HFTokenizer.from_pretrained('gpt2')
        # 设置pad token为eos token（GPT-2默认没有pad token）
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        # 暴露重要的token IDs
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    def encode(self, text, add_special_tokens=False, max_length=None, padding=False, truncation=False):
        # 使用HuggingFace tokenizer编码
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding='max_length' if padding else False,
            truncation=truncation,
            return_tensors=None
        )
        return encoded

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False):
        # 兼容HuggingFace风格的调用
        if isinstance(text, str):
            # 单个文本
            encoded = self.tokenizer(
                text,
                truncation=truncation,
                max_length=max_length,
                padding=padding,
                return_tensors=return_tensors
            )
        elif isinstance(text, list):
            # 批量文本
            encoded = self.tokenizer(
                text,
                truncation=truncation,
                max_length=max_length,
                padding=padding,
                return_tensors=return_tensors
            )
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")
        
        return encoded

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
