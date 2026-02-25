import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    TQDM_AVAILABLE = False
try:
    from sklearn import metrics
    SKLEARN_AVAILABLE = True
except Exception:
    metrics = None
    SKLEARN_AVAILABLE = False

from CrossAttention_model import CrossAttentionReID
from Dataloader import make_pair_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Cross-Attention ReID")
    parser.add_argument("--Dataset_name", type=str, default="Mars", help="Dataset key in factory")
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrained_path", type=str, default="jx_vit_base_p16_224-80ecf9dd.pth")
    parser.add_argument("--save_dir", type=str, default="checkpoints_baseline")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, use_tqdm: bool = True):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_probs = []

    iterator = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False) if (use_tqdm and TQDM_AVAILABLE) else loader
    for v1, v2, labels in iterator:
        v1 = v1.to(device)
        v2 = v2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(v1, v2)  # [B,1]
        loss = criterion(logits.view(-1), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        probs = torch.sigmoid(logits.detach()).view(-1)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()

        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

        if use_tqdm and TQDM_AVAILABLE:
            batch_acc = (preds == labels).float().mean().item()
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()
    # Debug print: training label distribution
    y_true_t = torch.cat(all_labels)
    pos = int(y_true_t.sum().item())
    neg = int(y_true_t.numel() - pos)
    print(f"[TRAIN] labels: pos={pos}, neg={neg}, ratio={y_true_t.float().mean().item():.4f}")
    if SKLEARN_AVAILABLE:
        try:
            auc = metrics.roc_auc_score(y_true, y_score)
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
    return epoch_loss, epoch_acc, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch: int, use_tqdm: bool = True):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_probs = []

    iterator = tqdm(loader, desc=f"Val   Epoch {epoch}", leave=False) if (use_tqdm and TQDM_AVAILABLE) else loader
    for v1, v2, labels in iterator:
        v1 = v1.to(device)
        v2 = v2.to(device)
        labels = labels.to(device)

        logits = model(v1, v2)
        loss = criterion(logits.view(-1), labels)

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()

        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

        if use_tqdm and TQDM_AVAILABLE:
            batch_acc = (preds == labels).float().mean().item()
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()
    if SKLEARN_AVAILABLE:
        try:
            auc = metrics.roc_auc_score(y_true, y_score)
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
    return epoch_loss, epoch_acc, auc


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    model = CrossAttentionReID(
        img_size=(args.img_height, args.img_width),
        embed_dim=768,
        num_heads=12,
        stride_size=16,
        camera_num=6,
        pretrained_path=args.pretrained_path,
    ).to(device)

    train_loader, val_loader = make_pair_dataloader(
        Dataset_name=args.Dataset_name,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, use_tqdm=True)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device, epoch, use_tqdm=True)

        elapsed = time.time() - start
        print(
            f"[Epoch {epoch:03d}] "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.save_dir, f"baseline_best_auc_{best_auc:.4f}.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "best_auc": best_auc,
            }, save_path)
            print(f"Saved best model to {save_path}")

    


if __name__ == "__main__":
    main()

