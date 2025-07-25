import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import os
import heapq
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from models.GPT import GPT, GPTConfig
from utils.logging_utils import (
    init_logging,
    log_train_metrics,
    log_val_metrics,
    log_attention,
    log_activations,
    finish_logging,
)
from utils.arg_utils import get_args
from utils.lr_scaler import get_linear_warmup_cosine_scheduler
from utils.data_loader import TinyStoryDataset
from utils.tokenizer import GPT2Tokenizer
from utils.params_calculator import calculate_gpt_params


class Trainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=os.path.join(args.log_dir, args.model_name))
        self.device = self.accelerator.device

        config = GPTConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )
        self.config = config
        self.model = GPT(config)
        
        self.tokenizer = GPT2Tokenizer()
        train_dataset = TinyStoryDataset(
            data_path=args.train_file,
            tokenizer=self.tokenizer,
            block_size=args.max_seq_len
        )
        val_dataset = TinyStoryDataset(
            data_path=args.val_file,
            tokenizer=self.tokenizer,
            block_size=args.max_seq_len
        )
        
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, collate_fn=self.collate_fn
        )

        self.num_training_steps = (len(self.train_dataloader) // args.gradient_accumulation_steps) * args.epochs
        self.num_warmup_steps = args.warmup_steps if args.warmup_steps else int(0.1 * self.num_training_steps)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = get_linear_warmup_cosine_scheduler(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            max_lr=config.learning_rate,
            min_lr=args.min_learning_rate
        )

        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler
        )
        
        self.best_ckpts = []
        self.ckpt_dir = os.path.join(args.log_dir, args.model_name, "ckpts")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        init_logging(log_dir=os.path.join(args.log_dir, args.model_name))

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-length sequences.
        Pads sequences to the maximum length in the batch.
        """
        # 从batch中提取input_ids和labels
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # 使用pad_sequence进行padding，batch_first=True表示batch维度在前
        # 使用 tokenizer 的 pad_token_id 进行填充
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100会被CrossEntropyLoss忽略
        
        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded
        }

    def loss_fn(self, outputs, labels):
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))

    def train(self):
        total_steps = 0
        for epoch in range(self.args.epochs):
            self.model.train()
            
            # 初始化epoch统计
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0
            epoch_steps = 0
            
            progress_bar = tqdm.tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}", disable=not self.accelerator.is_local_main_process)
            for batch_idx, batch in enumerate(progress_bar):
                should_log = total_steps % self.args.logging_steps == 0
                
                # 对输入进行安全性检查
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                
                # 打印第一个批次的调试信息
                if total_steps == 0 and batch_idx == 0 and self.accelerator.is_local_main_process:
                    print("\n--- 调试信息 ---")
                    print(f"input_ids 形状: {input_ids.shape}")
                    print(f"labels 形状: {labels.shape}")
                    print(f"input_ids 最小值: {input_ids.min().item()}, 最大值: {input_ids.max().item()}")
                    print(f"labels 最小值: {labels.min().item()}, 最大值: {labels.max().item()}")
                    print(f"tokenizer.vocab_size: {self.tokenizer.vocab_size}")
                    print(f"config.vocab_size: {self.config.vocab_size}")
                    print("---------------\n")
                
                # 确保所有token ID在有效范围内
                if input_ids.max() >= self.config.vocab_size:
                    print(f"警告: 输入中包含无效的token ID: max={input_ids.max().item()}, vocab_size={self.config.vocab_size}")
                    input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
                
                outputs = self.model(
                    input_ids, 
                    output_attentions=should_log, 
                    output_activations=should_log
                )
                loss = self.loss_fn(outputs['logits'], labels)
                loss = loss / self.args.gradient_accumulation_steps  # 缩放损失
                
                # 累积epoch统计
                with torch.no_grad():
                    preds = torch.argmax(outputs['logits'], dim=-1)
                    labels = batch["labels"]
                    mask = labels != -100
                    correct = (preds[mask] == labels[mask]).sum()
                    total_tokens = mask.sum()
                    
                    # 累积到epoch统计中
                    epoch_train_loss += loss.item() * self.args.gradient_accumulation_steps
                    epoch_train_correct += correct.item()
                    epoch_train_total += total_tokens.item()
                
                self.accelerator.backward(loss)

                # 梯度累积
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度裁剪并获取梯度范数
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                    # 手动计算梯度范数
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    total_steps += 1
                    epoch_steps += 1
                    
                    if self.accelerator.is_main_process and should_log:
                        # 计算准确率
                        preds = torch.argmax(outputs['logits'], dim=-1)
                        labels = batch["labels"]
                        mask = labels != -100  # 假设-100是忽略的token
                        correct = (preds[mask] == labels[mask]).sum()
                        total_tokens = mask.sum()
                        
                        # 收集所有进程的数据
                        gathered_correct = self.accelerator.gather(correct)
                        gathered_tokens = self.accelerator.gather(total_tokens)
                        train_acc = gathered_correct.sum().item() / gathered_tokens.sum().item()

                        avg_loss = self.accelerator.gather(loss * self.args.gradient_accumulation_steps).mean().item()  # 恢复原始损失
                        perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')
                        
                        log_train_metrics(
                            step=total_steps,
                            train_loss=avg_loss,
                            train_acc=train_acc,
                            lr=self.scheduler.get_last_lr()[0],
                            perplexity=perplexity,
                            grad_norm=grad_norm
                        )

                # 定期评估
                if hasattr(self.args, 'eval_steps') and total_steps % self.args.eval_steps == 0 and total_steps > 0:
                    if self.accelerator.is_main_process:
                        val_loss, val_acc = self.evaluate(epoch, log_metrics=False)  # 不在evaluate内部记录metrics
                        log_val_metrics(
                            step=total_steps,
                            val_loss=val_loss,
                            val_acc=val_acc
                        )
                        self.save_best_ckpt(val_loss, epoch)

            # 每个epoch结束后收集并打印训练统计
            if self.accelerator.is_main_process:
                # 计算epoch平均值
                avg_train_loss = epoch_train_loss / epoch_steps if epoch_steps > 0 else 0.0
                avg_train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0.0
                
                # 进行验证
                val_loss, val_acc = self.evaluate(epoch, log_metrics=False)  # 不在evaluate内部记录metrics
                
                # 打印epoch结果
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.args.epochs} Summary:")
                print(f"{'='*60}")
                print(f"Training   - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                print(f"{'='*60}\n")
                
                # 只在epoch结束时记录一次metrics
                log_val_metrics(
                    step=total_steps,
                    val_loss=val_loss,
                    val_acc=val_acc
                )
                self.save_best_ckpt(val_loss, epoch)
        
        # 在训练结束后记录最终的attention和activation
        if self.accelerator.is_main_process:
            self.log_final_artifacts()

        finish_logging()

    def log_final_artifacts(self):
        """获取一个验证批次，记录其attention和activation"""
        self.model.eval()
        # 确保在主进程上获取数据
        final_batch = next(iter(self.val_dataloader))
        # 将数据移动到正确的设备
        final_batch = {k: v.to(self.device) for k, v in final_batch.items()}

        with torch.no_grad():
            outputs = self.accelerator.unwrap_model(self.model)(
                final_batch["input_ids"],
                output_attentions=True,
                output_activations=True
            )
        
        if 'attentions' in outputs and outputs['attentions'] is not None:
            for i, attn in enumerate(outputs['attentions']):
                log_attention({f'final_layer_{i}': attn}, 'final')
        
        if 'activations' in outputs and outputs['activations'] is not None:
            for i, act_dict in enumerate(outputs['activations']):
                log_activations({f'final_layer_{i}_{name}': act for name, act in act_dict.items()}, 'final')
        
        self.accelerator.print("Final attention and activation artifacts have been logged.")

    def evaluate(self, epoch, log_metrics=True):
        # 验证
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_steps = 0
        with torch.no_grad():
            progress_bar = tqdm.tqdm(self.val_dataloader, desc="Evaluating", disable=not self.accelerator.is_local_main_process)
            for batch in progress_bar:
                outputs = self.model(batch["input_ids"])
                loss = self.loss_fn(outputs['logits'], batch["labels"])
                total_loss += self.accelerator.gather(loss).mean().item()
                
                # 计算准确率
                preds = torch.argmax(outputs['logits'], dim=-1)
                labels = batch["labels"]
                mask = labels != -100
                correct = (preds[mask] == labels[mask]).sum()
                
                total_correct += self.accelerator.gather(correct).sum().item()
                total_tokens += self.accelerator.gather(mask.sum()).sum().item()
                total_steps += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        # 只有在log_metrics=True时才记录验证指标（移除内部的metrics记录）
        # 这样避免了重复记录的问题
            
        return avg_loss, avg_acc

    def save_best_ckpt(self, val_loss, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f"model_epoch{epoch}_valloss{val_loss:.4f}.pth")
        if len(self.best_ckpts) < 2:
            self.accelerator.save(self.accelerator.unwrap_model(self.model).state_dict(), ckpt_path)
            heapq.heappush(self.best_ckpts, (-val_loss, ckpt_path))
        else:
            worst_loss, worst_path = max(self.best_ckpts)
            if val_loss < -worst_loss:
                self.accelerator.save(self.accelerator.unwrap_model(self.model).state_dict(), ckpt_path)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                self.best_ckpts.remove((worst_loss, worst_path))
                heapq.heappush(self.best_ckpts, (-val_loss, ckpt_path))
        while len(self.best_ckpts) > 2:
            _, pop_path = heapq.heappop(self.best_ckpts)
            if os.path.exists(pop_path):
                os.remove(pop_path)


def main():
    args = get_args()
    trainer = Trainer(args)
    
    # 计算并打印参数量
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    total_params = calculate_gpt_params(
        unwrapped_model.config.vocab_size, 
        unwrapped_model.config.d_model, 
        unwrapped_model.config.n_layers, 
        unwrapped_model.config.n_heads, 
        unwrapped_model.config.d_ff, 
        unwrapped_model.config.max_seq_len
    )
    if trainer.accelerator.is_main_process:
        print(f"Training model: {args.model_name}")
        print(f"Total parameters (calculated): {total_params/1e6:.2f}M")

    trainer.train()

if __name__ == "__main__":
    main()
