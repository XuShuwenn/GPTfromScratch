import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GPT Training Arguments")
    
    # 模型结构参数
    parser.add_argument('--model_name', type=str, default='GPT2-8M', help='Name of the model')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256, help='Model hidden size')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feedforward hidden size')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Peak learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--lr_decay_steps', type=int, default=1000, help='LR decay step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='LR decay rate')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup steps (auto-calculate if None)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # 日志和评估
    parser.add_argument('--logging_steps', type=int, default=50, help='Logging interval (steps)')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation interval (steps)')
    
    # 数据路径
    parser.add_argument('--train_file', type=str, default='data/tinystories/train', help='Training data file')
    parser.add_argument('--val_file', type=str, default='data/tinystories/validation', help='Validation data file')
    
    # 保存路径
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and checkpoints')
    
    args = parser.parse_args()
    return args

