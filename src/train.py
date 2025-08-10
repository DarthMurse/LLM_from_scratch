import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
import pandas as pd
import os
import time

from tokenizer import BPETokenizer
from dataset import TextDataset, DataConfig
from optimizer import AdamW, cosine_lr_schedule, gradient_clipping
from model import Transformer, ModelConfig, CrossEntropyLoss

@dataclass
class TrainConfig:
    batch_size: int = 64
    resume: bool = False
    # Optimizer paramters
    lr = 4e-4
    weight_decay = 0.1
    # lr scheduler parameters
    tw: int = 2000
    tc: int = 200000
    # checkpointing parameters
    save_dir: str = "../out/"
    name: str = "TinyStories_22M"
    save_iter: int = 50000
    log_iter: int = 10
    eval_iter: int = 1000
    # gradient clipping
    clip_threshold = 1.0

def get_model_size(model):
    result = 0
    for p in model.parameters():
        result += p.numel()
    return result

def train_single(model, optimizer, train_loader, val_loader, config: TrainConfig, device):
    save_base_dir = config.save_dir + config.name + '/'
    if not os.path.exists(save_base_dir):
        os.mkdir(save_base_dir)
    train_logger = {"iter": [], "loss": []}
    eval_logger = {"iter": [], "loss": []}
    loss_func = CrossEntropyLoss()
    t = 0
    total_step = len(train_loader)

    if config.resume:
        ckpt_name = sorted(os.listdir(save_base_dir))[-1]
        ckpt = torch.load(save_base_dir + ckpt_name)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["model"])
        config.__dict__.update(ckpt["config"])
        t = ckpt["iter"]

    whole_time = time.time()
    start_time = time.time()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_func(logits.flatten(end_dim=-2), y.flatten())

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), config.clip_threshold)
        cosine_lr_schedule(optimizer, t, config.lr, config.lr / 10, config.tw, config.tc)
        optimizer.step()
        
        if t % config.eval_iter == 0:
            val_loss = validate_single(model, val_loader, config, device)
            print(f"Iter {t}/{total_step}, time: {time.time()-start_time}s, train_loss: {loss.item()}, val_loss: {val_loss.item()}")
            train_logger["iter"].append(t)
            eval_logger["iter"].append(t)
            train_logger["loss"].append(loss.item())
            eval_logger["loss"].append(val_loss.item())
            start_time = time.time()
        elif t % config.log_iter == 0:
            print(f"Iter {t}/{total_step}, time: {time.time()-start_time}s, train_loss: {loss.item()}")
            train_logger["iter"].append(t)
            train_logger["loss"].append(loss.item())
            start_time = time.time()

        if t % config.save_iter == 0:
            save_path = save_base_dir + f"iter_{t:06d}.ckpt"
            save_checkpoint(model, optimizer, t, save_path, config)
            pd.DataFrame(train_logger).to_csv(save_base_dir + "info_train.csv")
            pd.DataFrame(eval_logger).to_csv(save_base_dir + "info_eval.csv")
        t += 1
    
    whole_duration = time.time() - whole_time
    val_loss = validate_single(model, val_loader, config, device)
    print(f"Training complete! Total iter {t}, total time: {whole_duration}s, final val_loss: {val_loss.item()}")
    save_path = save_base_dir + f"iter_{t:06d}.ckpt"
    save_checkpoint(model, optimizer, t, save_path, config)

def validate_single(model, val_loader, config: TrainConfig, device):
    avg_loss = 0
    count = 0
    with torch.no_grad():
        loss_func = CrossEntropyLoss()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_func(model(x).flatten(end_dim=-2), y.flatten())
            avg_loss = (count * avg_loss + x.shape[0] * loss) / (count + x.shape[0])
            count += x.shape[0]
    return avg_loss

def save_checkpoint(model, optimizer, i, save_path, config: TrainConfig):
    checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.__dict__,
            "iter": i
            }
    torch.save(checkpoint, save_path)

def main():
    device = "cuda:1"
    train_config = TrainConfig()
    model_config = ModelConfig()
    model = Transformer(model_config)
    print(f"Model initialized with {get_model_size(model)} parameters.")
    data_config = DataConfig(seq_len=257, batch_size=64)
    data_time = time.time()
    train_set = TextDataset("../data/tiny_story_train.pth", data_config)
    val_set = TextDataset("../data/tiny_story_valid.pth", data_config)
    data_duration = time.time() - data_time
    print(f"Dataset loading takes {data_duration} seconds ..")
    train_loader = DataLoader(train_set, batch_size=train_config.batch_size)
    val_loader = DataLoader(val_set, batch_size=train_config.batch_size)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    train_single(model, optimizer, train_loader, val_loader, train_config, device)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
