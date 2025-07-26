import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 模型和数据集配置
MODEL_NAMES = ['GPT2-14M', 'GPT2-29M', 'GPT2-49M']
DATASET_NAME = 'tinystories'
# 定义要可视化的指标
METRICS_TO_PLOT = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'perplexity', 'grad_norm']
# 统一的图表保存目录
SAVE_DIR = os.path.join('visualize', 'metrics')

def clean_and_prepare_data(df, metric_name):
    """清理和准备数据用于可视化"""
    if metric_name not in df.columns or 'step' not in df.columns:
        return None
    
    df_clean = df.dropna(subset=[metric_name]).copy()
    df_clean = df_clean.sort_values('step').drop_duplicates(subset=['step'], keep='last')
    
    if metric_name == 'perplexity':
        # 过滤掉过大的困惑度值以便于可视化
        df_clean = df_clean[df_clean['perplexity'] < 1000]
    
    return df_clean if len(df_clean) > 0 else None

def plot_combined_metric(metric_name, models_data, save_path):
    """为单个指标绘制包含多个模型曲线的图表"""
    plt.figure(figsize=(12, 7))
    
    # 为不同模型设置不同的颜色和标记
    colors = plt.cm.viridis(np.linspace(0, 1, len(MODEL_NAMES)))
    markers = ['o', 's', '^', 'd', 'v', 'p', '*']
    
    for i, (model_name, df_clean) in enumerate(models_data.items()):
        plt.plot(df_clean['step'], df_clean[metric_name], 
                 label=model_name,
                 linewidth=2, 
                 marker=markers[i % len(markers)], 
                 markersize=4, 
                 alpha=0.8,
                 color=colors[i])

    # 根据指标类型设置图表属性
    metric_title = metric_name.replace('_', ' ').title()
    if 'loss' in metric_name or 'perplexity' in metric_name:
        plt.yscale('log')
        plt.ylabel(f'{metric_title} (log scale)')
    elif 'acc' in metric_name:
        plt.ylim(0, max(1.0, plt.ylim()[1])) # 保证Y轴至少到1.0
        plt.ylabel(metric_title)
    else:
        plt.ylabel(metric_title)

    plt.title(f'{metric_title} Comparison across Models ({DATASET_NAME})', fontsize=16)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(title="Models", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {metric_name}: 图表已保存至 {save_path}")

def main():
    """主函数，执行所有可视化任务"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 清理旧的可视化文件
    print(f"清理旧的图表文件于: {SAVE_DIR}")
    for old_file in os.listdir(SAVE_DIR):
        if old_file.endswith('.png'):
            os.remove(os.path.join(SAVE_DIR, old_file))

    # 按指标进行循环
    for metric in METRICS_TO_PLOT:
        print(f"\n正在处理指标: {metric}")
        
        models_data = {}
        # 为当前指标收集所有模型的数据
        for model_name in MODEL_NAMES:
            csv_path = os.path.join('logs', model_name, 'metrics.csv')
            if not os.path.exists(csv_path):
                print(f"  - 警告: 未找到 {model_name} 的数据文件 {csv_path}，跳过")
                continue
            
            df = pd.read_csv(csv_path)
            df_clean = clean_and_prepare_data(df, metric)
            
            if df_clean is not None:
                models_data[model_name] = df_clean
                print(f"  - 已加载 {model_name} 的数据，数据点: {len(df_clean)}")
            else:
                print(f"  - {model_name} 无有效的 '{metric}' 数据")

        # 如果至少有一个模型有数据，则绘图
        if models_data:
            save_path = os.path.join(SAVE_DIR, f'{metric}_comparison.png')
            plot_combined_metric(metric, models_data, save_path)
        else:
            print(f"  ✗ 所有模型均无 '{metric}' 的有效数据，无法生成图表")

    print("\n所有模型的指标对比可视化完成！")

if __name__ == "__main__":
    main() 