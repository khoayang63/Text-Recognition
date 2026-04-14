import torch
import torch.nn as nn
from ocr_recognition import CRNN, get_loaders
from tqdm import tqdm
import math
import os

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels, label_lengths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        T, B, _ = outputs.size()
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

        loss = criterion(outputs, labels, input_lengths, label_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # update tqdm
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:06f}",
            # "lr_backbone": f"{optimizer.param_groups[0]['lr']:.1e}",
            # "lr_head": f"{optimizer.param_groups[1]['lr']:.1e}",
        })
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels, label_lengths in pbar:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            outputs = model(images)

            T, B, _ = outputs.size()
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            loss = criterion(outputs, labels, input_lengths, label_lengths)

            total_loss += loss.item()

            pbar.set_postfix({
                "val_loss": f"{loss.item():.4f}"
            })

    return total_loss / len(dataloader)

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # linear warmup
            return [
                base_lr * step / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
        


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }, path)  

def load_checkpoint(model, optimizer=None, scheduler=None, path=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print(f"Loaded checkpoint from epoch {epoch}")

    return epoch, loss

def main():
    hidden_size=256
    n_layers=3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(hidden_size=hidden_size, n_layers=n_layers).to(device)
    num_epochs=50
    train_loader, val_loader, _, _ = get_loaders()

    print("Using device:", device)


    # optimizer = torch.optim.AdamW([
    #     {"params": model.backbone.parameters(), "lr": 1e-4}, 
    #     {"params": model.proj.parameters(), "lr": 1e-3},
    #     {"params": model.gru.parameters(), "lr": 1e-3},
    #     {"params": model.head.parameters(), "lr": 1e-3},
    # ], weight_decay=1e-5)


    # criterion = nn.CTCLoss(
    #     blank=0,            # index của blank token
    #     reduction="mean",
    #     zero_infinity=True  # tránh NaN khi sequence lỗi
    # )

    # scheduler = WarmupCosineScheduler(
    #     optimizer,
    #     warmup_steps=warmup_steps,
    #     total_steps=total_steps
    # )    
    # total_steps = len(train_loader) * num_epochs
    # warmup_steps = int(0.1 * total_steps)  # 10% warmup

    scheduler_step_size = num_epochs * 0.5
    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True,
        reduction="mean",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=0.1
    )

    log_file = "train_log_resnet152_1.txt"
    best_loss = float("inf")

    # tạo file + header (chỉ chạy 1 lần)
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                path="checkpoints/best.pth"
            )
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")

if __name__ == "__main__":
    main()