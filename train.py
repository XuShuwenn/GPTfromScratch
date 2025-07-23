import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


from models.GPT import GPT, GPTConfig
from utils.wandb_utils import (
    init_wandb,
    log_metrics,
    log_artifact,
    log_grad_norm,
    log_weights,
    log_model_artifact,
    log_train_metrics,
)
from utils.arg_utils import get_args
from utils.lr_scaler import get_linear_warmup_cosine_scheduler
from utils.data_loader import TinyStoryDataset
from utils.tokenizer import GPT2Tokenizer


class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 config,
                 args):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_training_steps = (len(train_dataset) // args.batch_size) * args.epochs
        self.num_warmup_steps = int(0.1 * self.num_training_steps)  # 预热10%
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_linear_warmup_cosine_scheduler(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.1
        )
        self.best_loss = float("inf")
        self.best_model = None
        self.best_epoch = 0
        self.best_metrics = {}
        self.best_artifact = None
        self.best_artifact_name = "best_model.pth"
        self.wandb_run = None

    def train(self):
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        # 训练一个loop
        for epoch in range(self.args.epochs):
            for batch in self.train_dataset:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch["input_ids"])
                loss = self.loss_fn(outputs, batch["labels"])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()
                total_steps += 1
                if total_steps % self.args.logging_steps == 0:
                    self.log_metrics(total_loss / self.args.logging_steps, total_steps)
                    # INSERT_YOUR_CODE
                    avg_loss = total_loss / self.args.logging_steps
                    metrics = {"train/loss": avg_loss, "step": total_steps, "epoch": epoch}
                    log_metrics(metrics)
                    # 保存模型
                    if avg_loss < self.best_loss:
                        self.best_loss = avg_loss
                        self.best_model = self.model.state_dict()
                        self.best_epoch = epoch
                        self.best_metrics = metrics
                    total_loss = 0.0
                    
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        for batch in self.val_dataset:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch["input_ids"])
            loss = self.loss_fn(outputs, batch["labels"])
            total_loss += loss.item()
            total_steps += 1
        avg_loss = total_loss / total_steps
        metrics = {"val/loss": avg_loss, "step": total_steps, "epoch": epoch}
        log_metrics(metrics)
        return avg_loss


def main():
    from utils.arg_utils import get_args
    args = get_args()
    config = GPTConfig(**args.config)
    init_wandb(config)
    model = GPT(config)
    tokenizer = GPT2Tokenizer()
    train_dataset = TinyStoryDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        block_size=args.max_seq_len
    )
    val_dataset = TinyStoryDataset(
        file_path=args.val_file,
        tokenizer=tokenizer,
        block_size=args.max_seq_len
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        args=args
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()