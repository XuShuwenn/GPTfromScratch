import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GPT Training Arguments")
    # 模型结构参数
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256, help='Model hidden size')
    parser.add_argument('--n_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feedforward hidden size')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length')
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lr_decay_steps', type=int, default=1000, help='LR decay step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='LR decay rate')
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging interval (steps)')
    # 数据路径
    parser.add_argument('--train_file', type=str, default='data/train.txt', help='Training data file')
    parser.add_argument('--val_file', type=str, default='data/val.txt', help='Validation data file')
    # 保存路径
    parser.add_argument('--model_save_path', type=str, default='best_model.pth', help='Path to save model')
    # 其他参数可按需添加
    args = parser.parse_args()
    # 兼容 GPTConfig 的参数打包
    config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'max_seq_len': args.max_seq_len,
        'lr_decay_steps': args.lr_decay_steps,
        'lr_decay_rate': args.lr_decay_rate,
        'learning_rate': args.learning_rate,
        'logging_steps': args.logging_steps,
    }
    args.config = config
    return args
