import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_patterns(log_dir, save_dir):
    """
    统一的注意力可视化函数
    """
    attn_dir = os.path.join(log_dir, 'attention')
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(attn_dir):
        print(f"Attention directory {attn_dir} not found")
        return
    
    for fname in os.listdir(attn_dir):
        if fname.endswith('.npy'):
            attn = np.load(os.path.join(attn_dir, fname))
            if attn.ndim == 4:
                # [batch, head, seq, seq]
                batch_size, num_heads = attn.shape[:2]
                for b in range(min(batch_size, 2)):  # 限制批次数量
                    for h in range(min(num_heads, 4)):  # 限制头数量
                        attn_img = attn[b, h]
                        plt.figure(figsize=(6, 5))
                        plt.imshow(attn_img, aspect='auto', cmap='viridis')
                        plt.colorbar()
                        plt.title(f'{fname} | batch={b} head={h}')
                        plt.xlabel('Key')
                        plt.ylabel('Query')
                        plt.tight_layout()
                        outname = fname.replace('.npy', f'_b{b}_h{h}.png')
                        plt.savefig(os.path.join(save_dir, outname))
                        plt.close()
            elif attn.ndim == 3:
                # [batch, seq, seq]
                batch_size = attn.shape[0]
                for b in range(min(batch_size, 2)):
                    attn_img = attn[b]
                    plt.figure(figsize=(6, 5))
                    plt.imshow(attn_img, aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title(f'{fname} | batch={b}')
                    plt.xlabel('Key')
                    plt.ylabel('Query')
                    plt.tight_layout()
                    outname = fname.replace('.npy', f'_b{b}.png')
                    plt.savefig(os.path.join(save_dir, outname))
                    plt.close()
            else:
                # [seq, seq]
                plt.figure(figsize=(6, 5))
                plt.imshow(attn, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(fname)
                plt.xlabel('Key')
                plt.ylabel('Query')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, fname.replace('.npy', '.png')))
                plt.close()

if __name__ == "__main__":
    # 支持多个模型的注意力可视化
MODEL_NAMES = ['GPT2-14M', 'GPT2-29M', 'GPT2-49M']
    for model_name in MODEL_NAMES:
        log_dir = os.path.join('logs', model_name)
        save_dir = os.path.join('visualize', 'attentions', model_name)
        print(f"Processing attention visualization for {model_name}...")
        visualize_attention_patterns(log_dir, save_dir) 