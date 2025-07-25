import torch
import argparse
from models.GPT import GPT, GPTConfig
from utils.tokenizer import GPT2Tokenizer

class Inference:
    # ... (现有Inference类代码保持不变) ...
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
            # 在多输出字典中获取logits
            logits = outputs['logits'][:, -1, :] / temperature
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

def main():
    parser = argparse.ArgumentParser(description="GPT Inference Arguments")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model config to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Prompt to start generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of the generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k sampling')
    args = parser.parse_args()

    # 根据model_name加载配置
    # 这里需要一个机制来获取不同模型的配置，暂时用硬编码
    if args.model_name == 'GPT2-8M':
        config = GPTConfig(d_model=256, n_layers=12, n_heads=12, d_ff=1024)
    elif args.model_name == 'GPT2-28M':
        config = GPTConfig(d_model=512, n_layers=12, n_heads=16, d_ff=2048)
    elif args.model_name == 'GPT2-110M':
        config = GPTConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072)
    else:
        raise ValueError("Unknown model name")

    model = GPT(config)
    model.load_state_dict(torch.load(args.model_path))
    tokenizer = GPT2Tokenizer()
    
    inference = Inference(model, tokenizer, config)
    generated_text = inference.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("Generated Text:\n", generated_text)

if __name__ == "__main__":
    main()


