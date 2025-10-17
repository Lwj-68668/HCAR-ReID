# BCR-Focal损失函数详解

## 概述

BCR-Focal（Bi-directional Consistency Regularized Focal Loss）是一种专为层次跨视频注意力模型设计的损失函数，结合了Focal BCE损失和双向一致性正则化（Symmetric-KL），用于提升视频行人重识别（Video Re-ID）任务中的配对判别性能。

## 创新点

BCR-Focal损失函数的核心创新在于：

1. **Focal BCE强化困难样本**：通过Focal机制增强模型对困难样本的学习能力
2. **双向一致性正则化**：对正反拼接顺序的配对打分加入对称KL散度约束，确保跨视频注意力在两个方向上保持稳定、可泛化
3. **高效实现**：训练期零额外前向计算，同一次前向即可获取两个方向的logit
4. **推理兼容**：推理与重排序阶段无额外开销，完全兼容现有流程

## 损失函数组成

### 1. Focal BCE损失部分

Focal BCE损失是对标准二元交叉熵（BCE）的改进，主要解决样本不平衡和困难样本学习问题。

#### 数学表达

```
FocalBCE(p_t) = -α_t * (1-p_t)^γ * log(p_t)
```

其中：
- `p_t`：模型对正确类别的预测概率
- `α_t`：平衡正负样本的权重因子（α表示正样本权重，1-α表示负样本权重）
- `γ`：聚焦参数，控制对困难样本的关注程度

#### 实现细节

```python
class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        y = targets.float()
        # 基础 BCE (logits 形式，数值稳定)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
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
```

#### 特点

1. **动态权重调整**：通过`alpha`参数平衡正负样本的权重
2. **困难样本聚焦**：通过`gamma`参数增加对困难样本的关注度
3. **数值稳定**：使用logits形式计算，避免数值不稳定问题

### 2. 双向一致性正则化（Symmetric-KL）

双向一致性正则化确保跨视频注意力模型在两个方向上的预测保持一致。

#### 数学表达

```
SymKL(P_a, P_b) = KL(P_a || P_b) + KL(P_b || P_a)
```

其中：
- `P_a = sigmoid(logits_a)`：正向配对(v1,v2)的概率分布
- `P_b = sigmoid(logits_b)`：反向配对(v2,v1)的概率分布
- `KL(P || Q) = P * log(P/Q) + (1-P) * log((1-P)/(1-Q))`：KL散度

#### 实现细节

```python
def symmetric_kl_with_logits(logits_a, logits_b):
    """计算两个Bernoulli分布的对称KL散度"""
    pa = torch.sigmoid(logits_a).clamp(1e-6, 1-1e-6)
    pb = torch.sigmoid(logits_b).clamp(1e-6, 1-1e-6)
    
    # KL(pa||pb) + KL(pb||pa)
    kl_ab = pa * torch.log(pa / pb) + (1 - pa) * torch.log((1 - pa) / (1 - pb))
    kl_ba = pb * torch.log(pb / pa) + (1 - pb) * torch.log((1 - pb) / (1 - pa))
    
    return (kl_ab + kl_ba).mean()
```

#### 特点

1. **对称性约束**：确保(v1,v2)和(v2,v1)的预测一致性
2. **概率分布对齐**：通过KL散度度量两个方向预测的差异
3. **数值稳定**：使用clamp避免概率为0或1时的数值问题

### 3. 整体损失函数

BCR-Focal的总体损失函数是两部分的加权和：

```
总损失 = Focal_BCE损失 + symkl_w * 对称KL散度
```

其中`symkl_w`是双向一致性正则化的权重系数，控制着双向一致性约束的强度。

## 模型修改

### 1. 层次跨视频注意力模型修改

在`HierarchicalCrossAttention_model.py`中，需要修改前向传播函数以支持双向logits计算：

```python
def forward(self, video1: torch.Tensor, video2: torch.Tensor, return_rev_logit: bool = False) -> torch.Tensor:
    # ... 原有特征提取和注意力计算 ...
    
    # 原有正向配对计算
    pair_feat = torch.cat([video_feat1, video_feat2], dim=-1)
    logit = self.classifier_head(pair_feat).squeeze(-1)  # [B]
    
    if not return_rev_logit:
        return logit
    
    # 反向顺序（video2, video1）使用同一个head，零额外参数
    pair_feat_rev = torch.cat([video_feat2, video_feat1], dim=-1)
    logit_rev = self.classifier_head(pair_feat_rev).squeeze(-1)  # [B]
    return logit, logit_rev
```

### 2. 训练脚本修改

在`train_hierarchical.py`中，需要添加BCR-Focal相关参数和损失计算：

```python
# 添加参数
parser.add_argument("--use_focal", action="store_true", help="use focal BCE instead of vanilla BCE")
parser.add_argument("--focal_alpha", type=float, default=0.5)
parser.add_argument("--focal_gamma", type=float, default=2.0)
parser.add_argument("--symkl_w", type=float, default=0.10, help="weight of symmetric-KL regularizer")

# 训练循环中的损失计算
for v1, v2, labels in loader:
    v1, v2, labels = v1.to(device), v2.to(device), labels.to(device).float()
    
    optim.zero_grad()
    if args.symkl_w > 0:
        logits, logits_rev = model(v1, v2, return_rev_logit=True)
    else:
        logits = model(v1, v2, return_rev_logit=False)
    
    # 主损失：BCE 或 Focal
    base_loss = focal(logits, labels) if args.use_focal else bce(logits, labels)
    
    # 双向一致性
    if args.symkl_w > 0:
        loss_sym = symmetric_kl_with_logits(logits, logits_rev) * args.symkl_w
    else:
        loss_sym = logits.new_zeros(())
    
    loss = base_loss + loss_sym
    loss.backward()
    optim.step()
```

## 实验设置

### 消融实验设计

为了验证BCR-Focal的有效性，建议进行以下四组消融实验：

| 组别 | 损失函数配置 | 说明 |
|------|------------|------|
| A | BCE（原始） | 基线模型 |
| B | Focal（α=0.5, γ=2.0） | 仅使用Focal BCE |
| C | BCE + Sym-KL（w=0.10） | BCE加双向一致性 |
| D | Focal + Sym-KL（BCR-Focal） | 完整的BCR-Focal |

### 评估指标

- **配对分类指标**：Val AUC（验证集上的ROC曲线下面积）
- **Re-ID指标**：mAP（平均精度均值）和Rank-1（首名命中率）
- **统计可靠性**：3次不同随机种子或官方splits，报告均值±标准差

### 推荐训练命令

#### A | 基线（BCE）
```bash
python -u train_hierarchical.py \
  --Dataset_name Mars --seq_len 4 --batch_size 32 --epochs 50 --num_workers 4 \
  --pretrained_path jx_vit_base_p16_224-80ecf9dd.pth \
  --save_dir checkpoints_hierarchical_mars/bce
```

#### B | Focal
```bash
python -u train_hierarchical.py \
  --Dataset_name Mars --seq_len 4 --batch_size 32 --epochs 50 --num_workers 4 \
  --pretrained_path jx_vit_base_p16_224-80ecf9dd.pth \
  --use_focal --focal_alpha 0.5 --focal_gamma 2.0 \
  --save_dir checkpoints_hierarchical_mars/focal
```

#### C | BCE + Sym-KL
```bash
python -u train_hierarchical.py \
  --Dataset_name Mars --seq_len 4 --batch_size 32 --epochs 50 --num_workers 4 \
  --pretrained_path jx_vit_base_p16_224-80ecf9dd.pth \
  --symkl_w 0.10 \
  --save_dir checkpoints_hierarchical_mars/bce_symkl
```

#### D | BCR-Focal（推荐）
```bash
python -u train_hierarchical.py \
  --Dataset_name Mars --seq_len 4 --batch_size 32 --epochs 50 --num_workers 4 \
  --pretrained_path jx_vit_base_p16_224-80ecf9dd.pth \
  --use_focal --focal_alpha 0.5 --focal_gamma 2.0 \
  --symkl_w 0.10 \
  --save_dir checkpoints_hierarchical_mars/bcr_focal
```

## 应用与优势

### 应用场景

BCR-Focal损失函数特别适用于：
- 视频行人重识别（Video Re-ID）任务中的配对判别
- 跨视频注意力模型的训练
- 需要处理样本不平衡和困难样本的场景
- 对模型预测一致性有高要求的应用

### 技术优势

1. **高效性**：在单次前向传播中同时获取正向和反向logits，无需额外的计算开销
2. **灵活性**：通过参数开关可以单独使用Focal BCE或双向一致性正则化，也可以组合使用
3. **兼容性**：与现有模型架构完全兼容，推理阶段无需任何修改
4. **数值稳定性**：实现了数值稳定的logits形式计算，避免了概率计算中的数值问题
5. **训练稳定性**：双向一致性约束有助于稳定训练过程，减少方向性波动

### 预期效果

1. **困难样本处理**：Focal BCE能够增强模型对困难样本的学习能力
2. **预测一致性**：Sym-KL正则化确保模型在两个方向上的预测保持一致
3. **泛化能力**：整体损失函数有助于提升模型在复杂场景下的泛化性能
4. **重排序效果**：使用BCR-Focal训练的模型在重排序阶段能够提供更可靠的相似度融合

## 总结

BCR-Focal损失函数通过结合Focal BCE和双向一致性正则化，为层次跨视频注意力模型提供了一种高效、灵活且稳定的训练方案。其极简的实现方式（仅一个新文件加两处小改）使得它能够轻松集成到现有代码中，同时保持训练和推理流程的兼容性。在大规模、噪声较多的数据集（如MARS）上，BCR-Focal能够显著提升模型的判别能力和泛化性能，是视频行人重识别任务中的一种有效损失函数设计。