# pair_losses.py - BCR-Focal损失实现
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalBCEWithLogitsLoss(nn.Module):
    """简化版 Focal BCE：用于配对二分类（logits:[B], targets:{0,1}）
    
    Args:
        alpha: 正负样本权重平衡因子 (0.5表示平衡)
        gamma: 聚焦参数，越大越关注困难样本
        reduction: 损失聚合方式 ("mean" or "sum")
    """
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """数值稳定实现：以 BCE-with-logits 为基底，再乘以 Focal 因子。
        Args:
            logits: [B] 未经过sigmoid的预测值
            targets: [B] 二分类标签 {0, 1}
        Returns:
            focal loss值
        """
        y = targets.float()
        # 基础 BCE (logits 形式，数值稳定)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')  # [B]
        # p 与 pt（针对正样本取 p，负样本取 1-p）
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        # alpha_t（针对正样本取 alpha，负样本取 1-alpha）
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        # Focal 因子
        focal_factor = (1 - pt).pow(self.gamma)
        loss = alpha_t * focal_factor * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def symmetric_kl_with_logits(logits_a, logits_b):
    """计算两个Bernoulli分布的对称KL散度
    用于强制(v1,v2)和(v2,v1)配对的预测一致性
    
    Args:
        logits_a: [B] 正向配对的logits
        logits_b: [B] 反向配对的logits
    Returns:
        对称KL散度的均值
    """
    pa = torch.sigmoid(logits_a).clamp(1e-6, 1-1e-6)
    pb = torch.sigmoid(logits_b).clamp(1e-6, 1-1e-6)
    
    # KL(pa||pb) + KL(pb||pa)
    kl_ab = pa * torch.log(pa / pb) + (1 - pa) * torch.log((1 - pa) / (1 - pb))
    kl_ba = pb * torch.log(pb / pa) + (1 - pb) * torch.log((1 - pb) / (1 - pa))
    
    return (kl_ab + kl_ba).mean()
