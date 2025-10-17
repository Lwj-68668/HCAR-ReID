import argparse
import os
import time

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pair_losses import FocalBCEWithLogitsLoss, symmetric_kl_with_logits
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

from HierarchicalCrossAttention_model import HierarchicalCrossAttentionReID
from Dataloader import make_pair_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchical Cross-Attention ReID")
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
    parser.add_argument("--save_dir", type=str, default="checkpoints_hierarchical")
    parser.add_argument("--seed", type=int, default=42)
    
    # BCR-Focal参数
    parser.add_argument("--use_focal", action="store_true",
                        help="use focal BCE instead of vanilla BCE")
    parser.add_argument("--focal_alpha", type=float, default=0.5,
                        help="focal loss alpha parameter for positive/negative balance")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="focal loss gamma parameter for hard sample mining")
    parser.add_argument("--symkl_w", type=float, default=0.0,
                        help="weight of symmetric-KL regularizer for bidirectional consistency")
    
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
 
 
def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int, 
                   use_tqdm: bool = True, use_focal: bool = False, 
                   focal_criterion=None, symkl_w: float = 0.0):
    model.train()
    running_loss = 0.0
    running_base_loss = 0.0
    running_symkl_loss = 0.0
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
        
        # 判断是否需要计算双向一致性
        if symkl_w > 0:
            logits, logits_rev = model(v1, v2, return_rev_logit=True)
        else:
            logits = model(v1, v2, return_rev_logit=False)
        
        # 主损失：BCE或Focal
        if use_focal and focal_criterion is not None:
            base_loss = focal_criterion(logits, labels.float())
        else:
            base_loss = criterion(logits, labels.float())
        
        # 双向一致性损失
        if symkl_w > 0:
            loss_sym = symmetric_kl_with_logits(logits, logits_rev) * symkl_w
        else:
            loss_sym = logits.new_zeros(())
        
        # 总损失
        loss = base_loss + loss_sym
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_base_loss += base_loss.item() * labels.size(0)
        if symkl_w > 0:
            running_symkl_loss += loss_sym.item() * labels.size(0)
        total += labels.size(0)

        probs = torch.sigmoid(logits.detach())
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()

        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

        if use_tqdm and TQDM_AVAILABLE:
            batch_acc = (preds == labels).float().mean().item()
            postfix = {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
            if symkl_w > 0:
                postfix["symkl"] = f"{loss_sym.item():.4f}"
            iterator.set_postfix(postfix)

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
def evaluate(model, loader, criterion, device, epoch: int, use_tqdm: bool = True, 
            use_focal: bool = False, focal_criterion=None):
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
        
        # 使用focal或BCE损失
        if use_focal and focal_criterion is not None:
            loss = focal_criterion(logits, labels.float())
        else:
            loss = criterion(logits, labels.float())

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        probs = torch.sigmoid(logits)
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
    
    # Fix random seeds for reproducibility
    set_seed(args.seed)

    model = HierarchicalCrossAttentionReID(
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

    # 设置损失函数
    criterion = nn.BCEWithLogitsLoss()
    focal_criterion = None
    if args.use_focal:
        focal_criterion = FocalBCEWithLogitsLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"使用Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    
    if args.symkl_w > 0:
        print(f"使用对称KL正则化: weight={args.symkl_w}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            use_tqdm=True, use_focal=args.use_focal, 
            focal_criterion=focal_criterion, symkl_w=args.symkl_w
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device, epoch, 
            use_tqdm=True, use_focal=args.use_focal, 
            focal_criterion=focal_criterion
        )

        elapsed = time.time() - start
        print(
            f"[Epoch {epoch:03d}] "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.save_dir, f"hierarchical_best_auc_{best_auc:.4f}.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "best_auc": best_auc,
            }, save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
