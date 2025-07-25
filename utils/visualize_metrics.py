import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAMES = ['GPT2-14M', 'GPT2-29M', 'GPT2-49M']
DATASET_NAME = 'tinystories'

def clean_and_prepare_data(df, metric_name):
    """清理和准备数据用于可视化"""
    if metric_name not in df.columns or 'step' not in df.columns:
        return None
    
    # 删除指标为空的行
    df_clean = df.dropna(subset=[metric_name]).copy()
    
    # 按step排序，去除重复的step（保留最后一个）
    df_clean = df_clean.sort_values('step').drop_duplicates(subset=['step'], keep='last')
    
    # 特殊处理perplexity：过滤异常值
    if metric_name == 'perplexity':
        df_clean = df_clean[df_clean['perplexity'] < 1000]
    
    return df_clean if len(df_clean) > 0 else None

def plot_metric(df_clean, metric_name, save_path, model_name):
    """绘制单个指标的曲线图"""
    plt.figure(figsize=(10, 6))
    
    # 根据指标类型设置不同的样式
    if metric_name == 'perplexity':
        plt.plot(df_clean['step'], df_clean[metric_name], 
                linewidth=2, marker='o', markersize=3, alpha=0.8, color='red')
        plt.yscale('log')
        plt.ylabel('Perplexity (log scale)')
        plt.title(f'Perplexity Curve ({model_name}-{DATASET_NAME})')
    elif 'loss' in metric_name:
        plt.plot(df_clean['step'], df_clean[metric_name], 
                linewidth=2, marker='s', markersize=3, alpha=0.8, color='blue')
        plt.yscale('log')
        plt.ylabel(f'{metric_name.replace("_", " ").title()} (log scale)')
        plt.title(f'{metric_name.replace("_", " ").title()} Curve ({model_name}-{DATASET_NAME})')
    elif 'acc' in metric_name:
        plt.plot(df_clean['step'], df_clean[metric_name], 
                linewidth=2, marker='^', markersize=3, alpha=0.8, color='green')
        plt.ylabel(f'{metric_name.replace("_", " ").title()}')
        plt.ylim(0, 1.05)
        plt.title(f'{metric_name.replace("_", " ").title()} Curve ({model_name}-{DATASET_NAME})')
    elif metric_name == 'lr':
        plt.plot(df_clean['step'], df_clean[metric_name], 
                linewidth=2, marker='d', markersize=3, alpha=0.8, color='orange')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule ({model_name}-{DATASET_NAME})')
    elif metric_name == 'grad_norm':
        plt.plot(df_clean['step'], df_clean[metric_name], 
                linewidth=2, marker='v', markersize=3, alpha=0.8, color='purple')
        plt.ylabel('Gradient Norm')
        plt.title(f'Gradient Norm Curve ({model_name}-{DATASET_NAME})')
    else:
        plt.plot(df_clean['step'], df_clean[metric_name], 
                linewidth=2, marker='o', markersize=3, alpha=0.8)
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Curve ({model_name}-{DATASET_NAME})')
    
    plt.xlabel('Training Step')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

for model_name in MODEL_NAMES:
    CSV_PATH = os.path.join('logs', model_name, 'metrics.csv')
    SAVE_DIR = os.path.join('visualize', model_name, 'metrics')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"正在处理 {model_name} 模型的指标可视化...")
    
    if not os.path.exists(CSV_PATH):
        print(f"警告: 未找到 {CSV_PATH} 文件，跳过该模型")
        continue
    
    # 清理旧的可视化文件
    for old_file in os.listdir(SAVE_DIR):
        if old_file.endswith('.png'):
            os.remove(os.path.join(SAVE_DIR, old_file))
    
    # 读取数据
    df = pd.read_csv(CSV_PATH)
    print(f"读取到数据文件，原始行数: {len(df)}")
    
    # 定义要可视化的指标
    metrics_to_plot = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'perplexity', 'grad_norm']
    
    success_count = 0
    for metric in metrics_to_plot:
        df_clean = clean_and_prepare_data(df, metric)
        if df_clean is not None:
            save_path = os.path.join(SAVE_DIR, f'{metric}_curve.png')
            plot_metric(df_clean, metric, save_path, model_name)
            print(f"  ✓ {metric}: 生成成功，数据点: {len(df_clean)}")
            success_count += 1
        else:
            print(f"  ✗ {metric}: 无有效数据")
    
    print(f"模型 {model_name} 完成，成功生成 {success_count}/{len(metrics_to_plot)} 个指标图表")
    print()

print("所有模型的指标可视化完成！") 