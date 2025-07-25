import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_activation_distributions(log_dir, save_dir):
    """
    统一的激活值分布可视化函数
    """
    act_dir = os.path.join(log_dir, 'activations')
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(act_dir):
        print(f"Activations directory {act_dir} not found")
        return
        
    for fname in os.listdir(act_dir):
        if fname.endswith('.npy'):
            act = np.load(os.path.join(act_dir, fname))
            plt.figure(figsize=(7, 5))
            plt.hist(act.flatten(), bins=100, color='royalblue', alpha=0.7)
            plt.title(f'Activation Distribution: {fname}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname.replace('.npy', '.png')))
            plt.close()

if __name__ == "__main__":
    # 支持多个模型的激活值可视化
MODEL_NAMES = ['GPT2-14M', 'GPT2-29M', 'GPT2-49M']
    for model_name in MODEL_NAMES:
        log_dir = os.path.join('logs', model_name)
        save_dir = os.path.join('visualize', 'activations', model_name)
        print(f"Processing activation visualization for {model_name}...")
        visualize_activation_distributions(log_dir, save_dir) 