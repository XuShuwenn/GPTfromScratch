import os
import numpy as np
import matplotlib.pyplot as plt

ATTN_DIR = 'logs/attention'
SAVE_DIR = 'visualize/attention'
os.makedirs(SAVE_DIR, exist_ok=True)

for fname in os.listdir(ATTN_DIR):
    if fname.endswith('.npy'):
        attn = np.load(os.path.join(ATTN_DIR, fname))
        if attn.ndim == 4:
            # [batch, head, seq, seq]
            batch_size, num_heads = attn.shape[:2]
            for b in range(batch_size):
                for h in range(num_heads):
                    attn_img = attn[b, h]
                    plt.figure(figsize=(6, 5))
                    plt.imshow(attn_img, aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title(f'{fname} | batch={b} head={h}')
                    plt.xlabel('Key')
                    plt.ylabel('Query')
                    plt.tight_layout()
                    outname = fname.replace('.npy', f'_b{b}_h{h}.png')
                    plt.savefig(os.path.join(SAVE_DIR, outname))
                    plt.close()
        elif attn.ndim == 3:
            # [batch, seq, seq]
            batch_size = attn.shape[0]
            for b in range(batch_size):
                attn_img = attn[b]
                plt.figure(figsize=(6, 5))
                plt.imshow(attn_img, aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title(f'{fname} | batch={b}')
                plt.xlabel('Key')
                plt.ylabel('Query')
                plt.tight_layout()
                outname = fname.replace('.npy', f'_b{b}.png')
                plt.savefig(os.path.join(SAVE_DIR, outname))
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
            plt.savefig(os.path.join(SAVE_DIR, fname.replace('.npy', '.png')))
            plt.close() 