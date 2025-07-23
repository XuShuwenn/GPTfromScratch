import torch
from models.GPT import GPT, GPTConfig

class Inference:
    def __init__(self, 
                 model,
                 tokenizer, 
                 config,
                 device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, 
                 prompt,
                 max_length=100,
                 temperature=0.7,
                 top_k=20,
                 top_p=0.9,
                 eos_token_id=None):
        # 编码 prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids
        for _ in range(max_length):
            outputs = self.model(generated)
            logits = outputs[:, -1, :] / temperature
            # top-k 采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[0, torch.multinomial(probs, num_samples=1)]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return output_text


