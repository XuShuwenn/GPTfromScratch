import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, DatasetDict
import os

DATASET_LOCAL_DIR = "data/tinystories"

# 用于首次下载和缓存 TinyStories 数据集到本地
def download_tinystories(local_dir=DATASET_LOCAL_DIR):
    if not os.path.exists(local_dir):
        print(f"Downloading TinyStories to {local_dir} ...")
        dataset = load_dataset('roneneldan/TinyStories')
        dataset.save_to_disk(local_dir)
        print("Download and save complete.")
    else:
        print(f"TinyStories already exists at {local_dir}.")

class TinyStoryDataset(Dataset):
    def __init__(self, split, tokenizer, block_size=128, max_samples=None, return_attention_mask=False, local_dir=DATASET_LOCAL_DIR):
        # 优先从本地加载 TinyStories
        if os.path.exists(local_dir):
            dataset = load_from_disk(local_dir)[split]
        else:
            dataset = load_dataset('roneneldan/TinyStories', split=split)
        texts = [item['text'] for item in dataset]
        if max_samples:
            texts = texts[:max_samples]
        encodings = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=block_size,
            padding='max_length'
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings.get('attention_mask', None)
        self.return_attention_mask = return_attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        data = {
            'input_ids': self.input_ids[idx][:-1],
            'labels': self.input_ids[idx][1:]
        }
        if self.return_attention_mask and self.attention_mask is not None:
            data['attention_mask'] = self.attention_mask[idx][:-1]
        return data
