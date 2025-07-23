import os
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAMES = ['8M', '28M', '110M']
for model_name in MODEL_NAMES:
    ACT_DIR = os.path.join('logs', model_name, 'activations')
    SAVE_DIR = os.path.join('visualize', model_name, 'activations')
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(ACT_DIR):
        continue
    for fname in os.listdir(ACT_DIR):
        if fname.endswith('.npy'):
            act = np.load(os.path.join(ACT_DIR, fname))
            plt.figure(figsize=(7, 5))
            plt.hist(act.flatten(), bins=100, color='royalblue', alpha=0.7)
            plt.title(f'Activation Distribution: {fname}')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, fname.replace('.npy', '.png')))
            plt.close() 