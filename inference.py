import torch
import argparse
import os
from models.GPT import GPT, GPTConfig
from utils.tokenizer import GPT2Tokenizer

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
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, 
                 prompt,
                 max_length=100,
                 temperature=0.7,
                 top_k=20,
                 top_p=0.9,
                 eos_token_id=None,
                 repetition_penalty=1.1):
        """
        生成文本
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: top-p采样参数（暂未实现）
            eos_token_id: 结束标记ID
            repetition_penalty: 重复惩罚系数
        """
        print(f"Generating text with prompt: '{prompt}'")
        print(f"Parameters: max_length={max_length}, temperature={temperature}, top_k={top_k}, repetition_penalty={repetition_penalty}")
        
        # 编码 prompt
        try:
            if isinstance(prompt, str):
                input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
            else:
                input_ids = prompt
            input_ids = input_ids.to(self.device)
        except Exception as e:
            print(f"Error encoding prompt: {e}")
            return None
            
        generated = input_ids
        original_length = generated.size(1)
        
        print(f"Starting generation from {original_length} tokens...")
        
        for step in range(max_length):
            # 限制序列长度，避免超出模型能力
            if generated.size(1) > self.config.max_seq_len:
                # 保留最后 max_seq_len-1 个token，为新token留空间
                generated = generated[:, -(self.config.max_seq_len-1):]
            
            try:
                outputs = self.model(generated)
                # 在多输出字典中获取logits
                logits = outputs['logits'][:, -1, :].clone()
                
                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty
                
                # 应用温度
                logits = logits / temperature
                
                # top-k 采样
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否生成了结束标记
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    print(f"Generation stopped at step {step+1} due to EOS token")
                    break
                    
                # 简单的重复检测
                if step > 10:
                    recent_tokens = generated[0, -6:].tolist()
                    if len(set(recent_tokens)) == 1:  # 最近6个token都一样
                        print(f"Generation stopped at step {step+1} due to repetition")
                        break
                        
            except Exception as e:
                print(f"Error during generation at step {step}: {e}")
                break
        
        try:
            output_text = self.tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
            print(f"Generated {generated.size(1) - original_length} new tokens")
            return output_text
        except Exception as e:
            print(f"Error decoding generated text: {e}")
            return None

def load_model_and_config(args):
    """
    根据参数加载模型和配置
    """
    # 创建配置
    config = GPTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )
    
    print("Model Configuration:")
    print(f"  Vocab Size: {config.vocab_size}")
    print(f"  D Model: {config.d_model}")
    print(f"  N Layers: {config.n_layers}")
    print(f"  N Heads: {config.n_heads}")
    print(f"  D FF: {config.d_ff}")
    print(f"  Max Seq Len: {config.max_seq_len}")
    print(f"  Dropout: {config.dropout}")
    
    # 创建模型
    model = GPT(config)
    
    # 加载检查点
    print(f"Loading checkpoint from: {args.model_path}")
    try:
        # 检查检查点文件是否存在
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.model_path}")
            
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description="GPT Inference Script")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    # 生成参数
    parser.add_argument('--prompt', type=str, default="Once upon a time", 
                       help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, 
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, 
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=20, 
                       help='Top-k sampling parameter (0 = disabled)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='Repetition penalty (1.0 = no penalty, higher = less repetition)')
    # 模型结构参数
    parser.add_argument('--vocab_size', type=int, default=50257, 
                       help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=768, 
                       help='Model hidden size')
    parser.add_argument('--n_layers', type=int, default=12, 
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=12, 
                       help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, 
                       help='Feedforward hidden size')
    parser.add_argument('--max_seq_len', type=int, default=1024, 
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout rate')
    
    args = parser.parse_args()
    # 验证参数
    if args.temperature <= 0:
        print("Error: Temperature must be positive")
        return
    
    if args.max_length <= 0:
        print("Error: Max length must be positive")
        return
    
    print("=" * 50)
    print("GPT Model Inference")
    print("=" * 50)
    
    # 加载模型
    model, config = load_model_and_config(args)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # 初始化分词器
    tokenizer = GPT2Tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # 创建推理器
    inference = Inference(model, tokenizer, config)
    
    print("\n" + "=" * 50)
    print("Generating Text...")
    print("=" * 50)
    
    # 生成文本
    generated_text = inference.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.tokenizer.eos_token_id
    )
    
    if generated_text is not None:
        print("\n" + "=" * 50)
        print("Generated Text:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)
    else:
        print("Text generation failed.")

if __name__ == "__main__":
    main()
