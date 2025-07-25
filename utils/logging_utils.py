import logging
import csv
import os
import numpy as np
import torch

# 全局变量，由 init_logging 初始化
METRICS_CSV = ''
ACTIVATIONS_DIR = ''
ATTENTION_DIR = ''
csv_header_written = False

def init_logging(log_dir='logs'):
    """初始化日志记录器，设置文件路径和CSV表头。"""
    global METRICS_CSV, ACTIVATIONS_DIR, ATTENTION_DIR, csv_header_written
    
    METRICS_CSV = os.path.join(log_dir, 'metrics.csv')
    ACTIVATIONS_DIR = os.path.join(log_dir, 'activations')
    ATTENTION_DIR = os.path.join(log_dir, 'attention')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    os.makedirs(ATTENTION_DIR, exist_ok=True)
    
    csv_header_written = os.path.exists(METRICS_CSV)
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO, filemode='w')

    if not csv_header_written:
        with open(METRICS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'perplexity', 'grad_norm'])
        csv_header_written = True

def log_train_metrics(step, train_loss, train_acc, lr, perplexity, grad_norm):
    """记录训练指标到CSV和日志文件。"""
    with open(METRICS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step, train_loss, '', train_acc, '', lr, perplexity, grad_norm])
    logging.info(f"Step {step}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, lr={lr:.6f}, perplexity={perplexity:.4f}, grad_norm={grad_norm:.4f}")

def log_val_metrics(step, val_loss, val_acc):
    """通过读写CSV文件，更新对应step的验证指标。"""
    lines = []
    step_found = False
    try:
        with open(METRICS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            lines.append(header)
            for row in reader:
                if row and int(row[0]) == step:
                    # 更新 val_loss 和 val_acc
                    row[2] = f"{val_loss:.4f}"
                    row[4] = f"{val_acc:.4f}"
                    step_found = True
                lines.append(row)
        
        if not step_found:
            # 如果没有找到对应的step，说明是新的验证step，追加一行
            # 这种情况理论上不应该发生，因为验证总是在训练后
            lines.append([step, '', f"{val_loss:.4f}", '', f"{val_acc:.4f}", '', '', ''])

        with open(METRICS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(lines)
            
    except FileNotFoundError:
        # 如果文件不存在，直接创建并写入
        with open(METRICS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'perplexity', 'grad_norm'])
            writer.writerow([step, '', f"{val_loss:.4f}", '', f"{val_acc:.4f}", '', '', ''])

    logging.info(f"Step {step}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

def log_activations(activations_dict, step):
    """保存激活值到.npy文件。"""
    for name, act in activations_dict.items():
        path = os.path.join(ACTIVATIONS_DIR, f"{name.replace('.', '_')}_step{step}.npy")
        np.save(path, act.detach().cpu().numpy())
        logging.info(f"Step {step}: activation saved to {path}")

def log_attention(attention_dict, step):
    """保存注意力权重到.npy文件。"""
    for name, attn in attention_dict.items():
        path = os.path.join(ATTENTION_DIR, f"{name.replace('.', '_')}_step{step}.npy")
        np.save(path, attn.detach().cpu().numpy())
        logging.info(f"Step {step}: attention saved to {path}")

def finish_logging():
    """关闭日志记录器。"""
    logging.shutdown()
 