import os
import pandas as pd
import matplotlib.pyplot as plt

MODEL_NAMES = ['8M', '28M', '110M']
for model_name in MODEL_NAMES:
    CSV_PATH = os.path.join('logs', model_name, 'metrics.csv')
    SAVE_DIR = os.path.join('visualize', model_name, 'perplexity')
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        continue
    df = pd.read_csv(CSV_PATH)
    if 'perplexity' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df['step'], df['perplexity'], label='Perplexity')
        plt.xlabel('Step')
        plt.ylabel('Perplexity')
        plt.title(f'Perplexity Curve ({model_name})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'perplexity_curve.png'))
        plt.close()
    else:
        print(f'perplexity column not found in {CSV_PATH}') 