import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class TinyStoryDataset(Dataset):
    #TinyStory数据集类
    def __init__(self, data_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # 检查是否已经是预处理数据
        if "tokenized" in data_path:
            print(f"直接加载预处理数据: {data_path}")
            self.tokenized_datasets = load_from_disk(data_path)
            print(f"成功加载预处理数据，包含 {len(self.tokenized_datasets)} 个样本")
        else:
            # 原始数据处理逻辑
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
                    text = example["text"]  
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

                # 过滤掉长度不合适的样本
                # 最小长度为2（至少要有输入token和预测token）
                # 最大长度为block_size
                self.tokenized_datasets = self.tokenized_datasets.filter(
                    lambda example: 2 <= len(example["input_ids"]) <= block_size
                )

                print(f"正在将处理后的数据保存到 '{cache_path}'...")
                self.tokenized_datasets.save_to_disk(cache_path)
            print("数据已保存。")

    def __len__(self):
        return len(self.tokenized_datasets)

    def __getitem__(self, idx):
        input_ids = self.tokenized_datasets[idx]["input_ids"]
        
        # 标准的GPT训练方式：
        # input_ids: [tok0, tok1, tok2, ..., tokN-1]
        # labels:    [tok1, tok2, tok3, ..., tokN-1, EOS]
        # 
        # 但我们保持长度一致，让损失函数处理：
        # input_ids: [tok0, tok1, tok2, ..., tokN-1]  
        # labels:    [tok1, tok2, tok3, ..., tokN-1] (shift by 1)
        
        # 如果序列长度小于2，跳过（这种情况应该已经被过滤掉了）
        if len(input_ids) < 2:
            # 返回一个简单的样本作为fallback
            input_ids = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
            labels = [self.tokenizer.eos_token_id, -100]
        else:
            # 正确的标签：输入序列向右偏移1位
            labels = input_ids[1:] + [self.tokenizer.eos_token_id]
        
        # Padding到 block_size
        while len(input_ids) < self.block_size:
            input_ids.append(self.tokenizer.pad_token_id)
            labels.append(-100)  # pad token不参与损失计算
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }