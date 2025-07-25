import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class TinyStoryDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 缓存路径
        split_name = os.path.basename(data_path)
        base_dir = os.path.dirname(data_path)
        cache_path = os.path.join(base_dir, f"tokenized_{split_name}_bs{block_size}")

        try:
            print(f"正在尝试从 '{cache_path}' 加载缓存...")
            self.tokenized_datasets = load_from_disk(cache_path)
            print("成功从缓存加载。")
        except Exception:
            print("未找到缓存，正在处理全部原始数据...")
            raw_datasets = load_from_disk(data_path)

            def tokenize_function(example):
                text = example["story"]
                tokenized_output = self.tokenizer.encode(text)
                return {"input_ids": tokenized_output}

            # 全量分词处理
            self.tokenized_datasets = raw_datasets.map(
                tokenize_function,
                remove_columns=raw_datasets.column_names,
                batched=False,
                num_proc=os.cpu_count() // 2,
                desc="Running tokenizer on dataset"
            )

            # 过滤掉长度超过 block_size 的样本
            self.tokenized_datasets = self.tokenized_datasets.filter(
                lambda example: len(example["input_ids"]) <= block_size
            )

            print(f"正在将处理后的数据保存到 '{cache_path}'...")
            self.tokenized_datasets.save_to_disk(cache_path)
            print("数据已保存。")

    def __len__(self):
        return len(self.tokenized_datasets)

    def __getitem__(self, idx):
        item = self.tokenized_datasets[idx]
        input_ids = item['input_ids']
        labels = list(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }