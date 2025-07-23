import math
from torch.optim.lr_scheduler import LambdaLR

def get_linear_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, max_lr, min_lr=0.0):
    """
    线性预热+余弦衰减学习率调度器。
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        max_lr: 预热后最大学习率
        min_lr: 最小学习率
    Returns:
        LambdaLR 调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay * (max_lr - min_lr) / max_lr + min_lr / max_lr

    return LambdaLR(optimizer, lr_lambda)