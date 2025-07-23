import logging
import csv
import os
import numpy as np
import torch

def init_logging(log_dir='logs'):
    global METRICS_CSV, WEIGHTS_DIR, GRADS_DIR, ACTIVATIONS_DIR, ATTENTION_DIR
    METRICS_CSV = os.path.join(log_dir, 'metrics.csv')
    WEIGHTS_DIR = os.path.join(log_dir, 'weights')
    GRADS_DIR = os.path.join(log_dir, 'grads')
    ACTIVATIONS_DIR = os.path.join(log_dir, 'activations')
    ATTENTION_DIR = os.path.join(log_dir, 'attention')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(GRADS_DIR, exist_ok=True)
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    os.makedirs(ATTENTION_DIR, exist_ok=True)
    global csv_header_written
    csv_header_written = os.path.exists(METRICS_CSV)
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    if not csv_header_written:
        with open(METRICS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'perplexity'])
        csv_header_written = True

def log_train_metrics(step, train_loss, train_acc, lr, perplexity, log_dir='logs'):
    with open(os.path.join(log_dir, 'metrics.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step, train_loss, '', train_acc, '', lr, perplexity])
    logging.info(f"Step {step}: train_loss={train_loss}, train_acc={train_acc}, lr={lr}, perplexity={perplexity}")

def log_val_metrics(step, val_loss, val_acc, log_dir='logs'):
    with open(os.path.join(log_dir, 'metrics.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step, '', val_loss, '', val_acc, '', ''])
    logging.info(f"Step {step}: val_loss={val_loss}, val_acc={val_acc}")

def log_learning_rate(lr, step):
    logging.info(f"Step {step}: learning_rate={lr}")

def log_perplexity(perplexity, step):
    logging.info(f"Step {step}: perplexity={perplexity}")

def log_weights(model, step, log_dir='logs'):
    weights_dir = os.path.join(log_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    for name, param in model.named_parameters():
        np.save(os.path.join(weights_dir, f"{name.replace('.', '_')}_step{step}.npy"), param.detach().cpu().numpy())

def log_grads(model, step, log_dir='logs'):
    grads_dir = os.path.join(log_dir, 'grads')
    os.makedirs(grads_dir, exist_ok=True)
    for name, param in model.named_parameters():
        if param.grad is not None:
            np.save(os.path.join(grads_dir, f"{name.replace('.', '_')}_step{step}.npy"), param.grad.detach().cpu().numpy())
            logging.info(f"Step {step}: grad_dist_{name} saved.")

def log_activations(activations_dict, step, log_dir='logs'):
    activations_dir = os.path.join(log_dir, 'activations')
    os.makedirs(activations_dir, exist_ok=True)
    for name, act in activations_dict.items():
        np.save(os.path.join(activations_dir, f"{name.replace('.', '_')}_step{step}.npy"), act.detach().cpu().numpy())
        logging.info(f"Step {step}: activation_dist_{name} saved.")

def log_attention(attention_dict, step, log_dir='logs'):
    attention_dir = os.path.join(log_dir, 'attention')
    os.makedirs(attention_dir, exist_ok=True)
    for name, attn in attention_dict.items():
        np.save(os.path.join(attention_dir, f"{name.replace('.', '_')}_step{step}.npy"), attn.detach().cpu().numpy())
        logging.info(f"Step {step}: attention_dist_{name} saved.")

def finish_logging():
    logging.shutdown() 