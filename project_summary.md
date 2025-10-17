# VID-Trans-ReID-main 项目概览

## 项目结构
```text
VID-Trans-ReID-main/
└── Data/
    ├── MARS/                              # MARS 视频行人重识别数据集
    │   ├── bbox_train/                    # 训练集图像（625个行人）
    │   │   ├── 0000/
    │   │   │   ├── 0000C1T0001F001.jpg    # 命名规则：{ID}C{cam}T{track}F{frame}.jpg
    │   │   │   ├── 0000C1T0001F002.jpg    # - C{cam}: 摄像头编号（1~6）
    │   │   │   └── ...                    # - T{track}: 轨迹编号
    │   │   ├── 0002/                      # - F{frame}: 帧序号
    │   │   └── ...                        # 行人ID不连续（如 0000, 0002, ..., 1500）
    │   │
    │   ├── bbox_test/                     # 测试集图像（636个行人，与训练集身份无重叠）
    │   │   ├── 0000/
    │   │   └── ...
    │   │
    │   └── info/                          # 官方元信息文件
    │       ├── tracks_train_info.mat      # 训练轨迹元数据（ID、摄像头、帧范围等）
    │       ├── tracks_test_info.mat       # 测试轨迹元数据
    │       ├── query_IDX.mat              # ReID 查询/图库划分（用于评估）
    │       ├── train_name.txt             # 训练行人ID列表（文本）
    │       └── test_name.txt              # 测试行人ID列表（文本）
    │
    └── i-LIDS-VID/                        # i-LIDS-VID 视频行人重识别数据集
        └── i-LIDS-VID/
            ├── images/                    # 预处理后的行人图像（用于常规 ReID）
            │   ├── cam1/                  # 摄像头1视角
            │   │   ├── person001/
            │   │   │   ├── cam1_person001.png                 # 单帧代表图（部分行人）
            │   │   │   ├── cam1_person001_00317.png           # 多帧序列（带原始帧编号）
            │   │   │   └── ...
            │   │   └── ...
            │   │   └── person319/
            │   └── cam2/                  # 摄像头2视角（结构同 cam1）
            │       ├── person001/
            │       └── ...
            │
            ├── sequences/                 # 完整视频序列（用于时序建模，如 Transformer）
            │   ├── cam1/
            │   │   ├── person001/
            │   │   │   ├── cam1_person001_00317.png
            │   │   │   ├── cam1_person001_00318.png
            │   │   │   └── ...
            │   │   └── ...
            │   └── cam2/
            │       ├── person001/
            │       └── ...
            │
            └── train-test people splits/  # 标准训练/测试人员划分
                ├── readme.txt
                ├── train_test_splits_ilidsvid.mat
                ├── train_test_splits_prid.mat
                └── splits.json
├── Datasets/
│   ├── MARS_dataset.py
│   ├── PRID_dataset.py
│   └── iLDSVID.py
├── checkpoints_baseline/
│   ├── tb/
│   │   ├── events.out.tfevents.1755587657.LAPTOP-JQEOT4LR.8216.0
│   │   ├── events.out.tfevents.1755588814.LAPTOP-JQEOT4LR.9052.0
│   │   └── events.out.tfevents.1755589753.LAPTOP-JQEOT4LR.23788.0
│   ├── ... (4 more .pth files)
│   ├── baseline_best_auc_0.4468.pth
│   ├── baseline_best_auc_0.4726.pth
│   └── baseline_best_auc_0.5413.pth
├── checkpoints_hierarchical/
│   ├── ... (6 more .pth files)
│   ├── hierarchical_best_auc_0.4132.pth
│   ├── hierarchical_best_auc_0.5936.pth
│   └── hierarchical_best_auc_0.6525.pth
├── checkpoints_hierarchical_mars/
│   ├── focal/
│   ├── hierarchical_best_auc_0.9423.pth
│   └── hierarchical_best_auc_0.9726.pth
├── loss/
│   ├── center_loss.py
│   ├── softmax_loss.py
│   └── triplet_loss.py
├── losses/
├── ... (1 more .pth files)
├── BCR-Focal损失函数详解.md
├── CrossAttention_model.py
├── Dataloader.py
├── HierarchicalCrossAttention_model.py
├── Loss_fun.py
├── Mars_Main_Model.pth
├── VID_Test.py
├── VID_Trans_ReID.py
├── VID_Trans_model.py
├── iLIDSVID_best.pth
├── iLIDSVID_best_mAP.pth
├── pair_losses.py
├── requirements.txt
├── rerank_evaluate.py
├── train_baseline.py
├── train_hierarchical.py
├── utility.py
└── vit_ID.py
```

## 文件内容
### 文件: `CrossAttention_model.py`

```python
import torch
import torch.nn as nn
from typing import Optional

from vit_ID import TransReID


class CrossAttentionReID(nn.Module):
    """
    Baseline cross-video matching model:
    - Shared ViT backbone (TransReID) extracts per-frame token sequences
    - Bidirectional cross-video attention over flattened patch-token sequences (drop [CLS])
    - Small MLP classifier head outputs a single matching logit
    """

    def __init__(
        self,
        img_size=(256, 128),
        embed_dim: int = 768,
        num_heads: int = 12,
        stride_size: int = 16,
        camera_num: int = 6,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Shared backbone (no classification head required for baseline)
        self.backbone = TransReID(
            img_size=img_size,
            patch_size=16,
            stride_size=stride_size,
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            camera=camera_num,
            drop_path_rate=0.0,
        )

        if pretrained_path is not None:
            # Load pretrained ViT weights (TransReID handles pos_embed resize etc.)
            try:
                self.backbone.load_param(pretrained_path)
            except Exception as exc:
                print(f"[CrossAttentionReID] Warning: failed to load pretrained weights: {exc}")

        # One shared multi-head attention used in both directions
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)

        # Lightweight classifier head: concat([B,E],[B,E]) -> [B,1]
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim * 2, 1),
        )

    def _extract_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, S, C, H, W]
        returns tokens: [B, S, N, E]
        """
        bsz, seq_len, c, h, w = frames.shape
        frames = frames.view(bsz * seq_len, c, h, w)
        # TransReID requires camera labels; use zeros as placeholder for baseline
        cam_labels = torch.zeros(bsz * seq_len, dtype=torch.long, device=frames.device)
        tokens = self.backbone(frames, cam_label=cam_labels)  # [B*S, N, E]
        n_tokens = tokens.shape[1]
        tokens = tokens.view(bsz, seq_len, n_tokens, -1)
        return tokens

    def forward(self, video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
        """
        video1, video2: [B, S, C, H, W]
        returns logit: [B, 1]
        """
        # a) feature extraction per frame -> token sequences (include [CLS])
        tokens1 = self._extract_tokens(video1)  # [B, S, N, E]
        tokens2 = self._extract_tokens(video2)  # [B, S, N, E]

        # b) drop [CLS], keep patch tokens only
        patch_tokens1 = tokens1[:, :, 1:, :]  # [B, S, P, E]
        patch_tokens2 = tokens2[:, :, 1:, :]  # [B, S, P, E]

        # c) flatten temporal and spatial patch dims to one long sequence per video
        B, S, P, E = patch_tokens1.shape
        seq1 = patch_tokens1.reshape(B, S * P, E)  # [B, S*P, E]
        seq2 = patch_tokens2.reshape(B, S * P, E)  # [B, S*P, E]

        # d) bidirectional cross attention
        # i. transpose to (L, N, E) where L=S*P, N=B
        q1 = seq1.permute(1, 0, 2)
        k2v2 = seq2.permute(1, 0, 2)
        q2 = seq2.permute(1, 0, 2)
        k1v1 = seq1.permute(1, 0, 2)

        # ii. cross attention in both directions
        fused1, _ = self.cross_attention(query=q1, key=k2v2, value=k2v2)  # [L, B, E]
        fused2, _ = self.cross_attention(query=q2, key=k1v1, value=k1v1)  # [L, B, E]

        # iii. back to (B, L, E)
        fused1 = fused1.permute(1, 0, 2)
        fused2 = fused2.permute(1, 0, 2)

        # e) average pool along sequence (temporal x spatial patches)
        feat1 = fused1.mean(dim=1)  # [B, E]
        feat2 = fused2.mean(dim=1)  # [B, E]

        # f/g) concat & classify
        pair_feat = torch.cat([feat1, feat2], dim=-1)  # [B, 2E]
        logit = self.classifier_head(pair_feat)  # [B, 1]

        # h) return logit
        return logit
```

### 文件: `Dataloader.py`

```python
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
import math

from timm.data.random_erasing import RandomErasing
from utility import RandomIdentitySampler,RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

__factory = {
    'Mars':Mars,
    'iLIDSVID':iLIDSVID,
    'PRID':PRID
}


def _seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a deterministic seed.
    PyTorch sets torch.initial_seed() for each worker; we align numpy and python random with it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_collate_fn(batch):
    
    
    imgs, pids, camids,a= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(a, dim=0)

def val_collate_fn(batch):
    
    imgs, pids, camids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids_batch,  img_paths



def pair_collate_fn(batch):
    v1, v2, labels = zip(*batch)
    v1 = torch.stack(v1, dim=0)
    v2 = torch.stack(v2, dim=0)
    labels = torch.stack(labels, dim=0)
    return v1, v2, labels

def dataloader(Dataset_name):
    train_transforms = T.Compose([
            T.Resize([256, 128], interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            
            
        ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    

    dataset = __factory[Dataset_name]()
    train_set = VideoDataset_inderase(dataset.train, seq_len=4, sample='intelligent',transform=train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

   
    train_loader = DataLoader(train_set, batch_size=64,sampler=RandomIdentitySampler(dataset.train, 64,4),num_workers=4, collate_fn=train_collate_fn)
  
    q_val_set = VideoDataset(dataset.query, seq_len=4, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=4, sample='dense', transform=val_transforms)
    
    
    return train_loader, len(dataset.query), num_classes, cam_num, view_num,q_val_set,g_val_set



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class VideoPairDataset(Dataset):
    """Video Pair Dataset for Cross-Video Matching.
    Returns (video1_tensor, video2_tensor, label) where label in {1.0, 0.0}.
    Each video tensor has shape [seq_len, C, H, W].
    """
    def __init__(self, dataset, seq_len=8, transform=None):
        # dataset: list of (img_paths, pid, camid)
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform

        # group indices by pid -> list of tracklet indices
        self.pid_to_indices = {}
        for index, (_, pid, _) in enumerate(self.dataset):
            if pid in self.pid_to_indices:
                self.pid_to_indices[pid].append(index)
            else:
                self.pid_to_indices[pid] = [index]
        self.pids = list(self.pid_to_indices.keys())

    def __len__(self):
        # one epoch length aligned with number of tracklets for convenience
        return len(self.dataset)

    def _sample_video_frames(self, img_paths):
        """Sample a clip of length seq_len from img_paths and apply transform.
        Strategy matches VideoDataset 'random' sampling.
        """
        num = len(img_paths)
        frame_indices = range(num)
        rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.seq_len, len(frame_indices))
        indices = frame_indices[begin_index:end_index]
        if len(indices) < self.seq_len:
            indices = list(indices)
            indices.extend([indices[-1] for _ in range(self.seq_len - len(indices))])
        else:
            indices = list(indices)

        imgs = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)  # [seq_len, C, H, W]
        return imgs

    def _sample_video_frames_with_begin(self, img_paths, begin_index: int):
        """Sample a clip with a specified begin index, used to enforce distinct clips."""
        num = len(img_paths)
        frame_indices = range(num)
        begin_index = max(0, min(begin_index, max(0, len(frame_indices) - 1)))
        end_index = min(begin_index + self.seq_len, len(frame_indices))
        indices = frame_indices[begin_index:end_index]
        if len(indices) < self.seq_len:
            indices = list(indices)
            indices.extend([indices[-1] for _ in range(self.seq_len - len(indices))])
        else:
            indices = list(indices)

        imgs = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        return imgs

    def __getitem__(self, _index):
        # decide positive (1.0) or negative (0.0) with 50% probability
        is_positive = random.random() < 0.5

        if is_positive:
            # pick a pid with at least one tracklet
            pid = random.choice(self.pids)
            idxs = self.pid_to_indices[pid]
            if len(idxs) >= 2:
                i1, i2 = random.sample(idxs, 2)
            else:
                i1 = i2 = idxs[0]
            (img_paths1, _, _ ) = self.dataset[i1]
            if i1 != i2:
                (img_paths2, _, _ ) = self.dataset[i2]
                v1 = self._sample_video_frames(img_paths1)
                v2 = self._sample_video_frames(img_paths2)
            else:
                # same tracklet: take two distinct clips by enforcing different begin indices when possible
                num = len(img_paths1)
                if num > self.seq_len + 1:
                    rand_end = max(0, num - self.seq_len - 1)
                    begin1 = random.randint(0, rand_end)
                    begin2 = random.randint(0, rand_end)
                    if begin2 == begin1 and rand_end > 0:
                        begin2 = (begin1 + self.seq_len) % (rand_end + 1)
                    v1 = self._sample_video_frames_with_begin(img_paths1, begin1)
                    v2 = self._sample_video_frames_with_begin(img_paths1, begin2)
                else:
                    v1 = self._sample_video_frames(img_paths1)
                    v2 = self._sample_video_frames(img_paths1)
            label = torch.tensor(1.0, dtype=torch.float32)
            return v1, v2, label
        else:
            # pick two different pids
            if len(self.pids) >= 2:
                pid1, pid2 = random.sample(self.pids, 2)
            else:
                # degenerate case: fall back to same pid if only one exists
                pid1 = pid2 = self.pids[0]
            i1 = random.choice(self.pid_to_indices[pid1])
            i2 = random.choice(self.pid_to_indices[pid2])
            (img_paths1, _, _ ) = self.dataset[i1]
            (img_paths2, _, _ ) = self.dataset[i2]
            v1 = self._sample_video_frames(img_paths1)
            v2 = self._sample_video_frames(img_paths2)
            label = torch.tensor(0.0, dtype=torch.float32)
            return v1, v2, label

class QueryGalleryPairDataset(Dataset):
    """Validation Pair Dataset built from Query and Gallery sets.
    - Deterministically builds a balanced list of positive/negative pairs.
    - Uses centered sampling for stability across epochs.
    Returns (video1_tensor, video2_tensor, label) where label in {1.0, 0.0}.
    Each video tensor has shape [seq_len, C, H, W].
    """
    def __init__(self, query_set, gallery_set, seq_len=8, transform=None):
        self.query = query_set
        self.gallery = gallery_set
        self.seq_len = seq_len
        self.transform = transform

        # Build pid->indices maps
        self.q_pid_to_indices = {}
        for idx, (_, pid, _) in enumerate(self.query):
            self.q_pid_to_indices.setdefault(pid, []).append(idx)
        self.g_pid_to_indices = {}
        for idx, (_, pid, _) in enumerate(self.gallery):
            self.g_pid_to_indices.setdefault(pid, []).append(idx)

        # Build deterministic positive pairs: one gallery per query if exists
        self.pairs = []  # (q_idx, g_idx, label)
        pos_q_indices = []
        for q_idx, (_, q_pid, _) in enumerate(self.query):
            g_list = self.g_pid_to_indices.get(q_pid, [])
            if g_list:
                g_idx = g_list[0]
                self.pairs.append((q_idx, g_idx, 1.0))
                pos_q_indices.append(q_idx)

        # Build negative pairs: one per positive query with a different pid
        gallery_pids = list(self.g_pid_to_indices.keys())
        for q_idx in pos_q_indices:
            _, q_pid, _ = self.query[q_idx]
            neg_pid = None
            for pid in gallery_pids:
                if pid != q_pid:
                    neg_pid = pid
                    break
            if neg_pid is None:
                # fallback: skip if gallery has only one pid
                continue
            g_idx = self.g_pid_to_indices[neg_pid][0]
            self.pairs.append((q_idx, g_idx, 0.0))

    def __len__(self):
        return len(self.pairs)

    def _sample_video_frames_center(self, img_paths):
        """Deterministic, centered sampling for validation stability."""
        num = len(img_paths)
        frame_indices = range(num)
        if num <= self.seq_len:
            indices = list(frame_indices)
            indices.extend([indices[-1] for _ in range(self.seq_len - len(indices))])
        else:
            begin_index = (num - self.seq_len) // 2
            end_index = begin_index + self.seq_len
            indices = list(frame_indices)[begin_index:end_index]
        imgs = []
        for idx in indices:
            img_path = img_paths[int(idx)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        return imgs

    def __getitem__(self, index):
        q_idx, g_idx, label = self.pairs[index]
        (q_img_paths, _, _) = self.query[q_idx]
        (g_img_paths, _, _) = self.gallery[g_idx]
        v1 = self._sample_video_frames_center(q_img_paths)
        v2 = self._sample_video_frames_center(g_img_paths)
        label = torch.tensor(label, dtype=torch.float32)
        return v1, v2, label

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            targt_cam=[]
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                targt_cam.append(camid)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            targt_cam=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    targt_cam.append(camid)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, targt_cam,img_paths
            #return imgs_array, pid, int(camid),trackid


        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

            

        
class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        labels = []
        targt_cam=[]
        
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img , temp  = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
            targt_cam.append(camid)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        
        return imgs, pid, targt_cam ,labels
        


def make_pair_dataloader(Dataset_name, seq_len=4, batch_size=32, num_workers=4, seed: int = 42):
    train_transforms = T.Compose([
            T.Resize([256, 128], interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = __factory[Dataset_name]()

    train_pair_set = VideoPairDataset(dataset.train, seq_len=seq_len, transform=train_transforms)
    val_pair_set = QueryGalleryPairDataset(dataset.query, dataset.gallery, seq_len=seq_len, transform=val_transforms)

    # Deterministic generator for shuffle and transforms
    g = torch.Generator()
    g.manual_seed(seed)

    train_pair_loader = DataLoader(
        train_pair_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pair_collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_pair_loader = DataLoader(
        val_pair_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pair_collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_pair_loader, val_pair_loader
```

### 文件: `Datasets/MARS_dataset.py`

```python
from __future__ import print_function, absolute_import

from collections import defaultdict
from scipy.io import loadmat
import os.path as osp
import numpy as np


class Mars(object):
    """
    MARS
    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
   
    root  = osp.join('Data', 'MARS') # default dataset root inside project: Data/MARS
   
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0, ):
        self._check_before_run()
        
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        # Detect correct home dirs (handle accidental nested folders like bbox_train/bbox_train)
        self.train_home = 'bbox_train'
        self.test_home = 'bbox_test'
        nested_train = osp.join(self.root, 'bbox_train', 'bbox_train')
        nested_test = osp.join(self.root, 'bbox_test', 'bbox_test')
        if osp.exists(nested_train):
            self.train_home = osp.join('bbox_train', 'bbox_train')
        if osp.exists(nested_test):
            self.test_home = osp.join('bbox_test', 'bbox_test')

        train, num_train_tracklets, num_train_pids, num_train_imgs =           self._process_data(train_names, track_train, home_dir=self.train_home, relabel=True, min_seq_len=min_seq_len)

        video = self._process_train_data(train_names, track_train, home_dir=self.train_home, relabel=True, min_seq_len=min_seq_len)


        query, num_query_tracklets, num_query_pids, num_query_imgs =           self._process_data(test_names, track_query, home_dir=self.test_home, relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs =           self._process_data(test_names, track_gallery, home_dir=self.test_home, relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        # self.train_videos = video
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams=6
        self.num_query_cams=6
        self.num_gallery_cams=6
        self.num_train_vids=num_train_tracklets
        self.num_query_vids=num_query_tracklets
        self.num_gallery_vids=num_gallery_tracklets
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir is not None and ('bbox_train' in home_dir or 'bbox_test' in home_dir)
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        if not relabel: pid2label = {pid:int(pid) for label, pid in enumerate(pid_list)}
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            #if relabel: pid = pid2label[pid]
            pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))
                # if camid in video[pid] :
                #     video[pid][camid].append(img_paths)  
                # else:
                #     video[pid][camid] =  img_paths

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)

        assert home_dir is not None and ('bbox_train' in home_dir or 'bbox_test' in home_dir)
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                if camid in video[pid] :
                    video[pid][camid].extend(img_paths)  
                else:
                    video[pid][camid] =  img_paths
        return video
```

### 文件: `Datasets/PRID_dataset.py`

```python
from __future__ import print_function, absolute_import

from collections import defaultdict
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
import glob
def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

class PRID(object):
    """
    PRID
    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2
    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root  = "prid_2011"
    
    # root = './data/prid2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams=2
        self.num_query_cams=2
        self.num_gallery_cams=2
        self.num_train_vids=num_train_tracklets
        self.num_query_vids=num_query_tracklets
        self.num_gallery_vids=num_gallery_tracklets

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
```

### 文件: `Datasets/iLDSVID.py`

```python
from __future__ import print_function, absolute_import

from collections import defaultdict
from scipy.io import loadmat
import os.path as osp
import numpy as np
import json
import errno
import os
import tarfile
import glob
def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj
import urllib.request


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
        
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class iLIDSVID(object):
    """
    iLIDS-VID
    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2
    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    root = "./Data"
    #root = "/root/autodl-tmp/reid/transreid/Data"
    # root = '/mnt/scratch/1/pathak/data/iLIDS'
    # root = './data/ilids-vid'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
    data_dir = osp.join(root, 'i-LIDS-VID')
    split_dir = osp.join(root, 'i-LIDS-VID', 'train-test people splits')
    split_mat_path = osp.join(split_dir, 'train_test_splits_ilidsvid.mat')
    split_path = osp.join(root, 'i-LIDS-VID', 'splits.json')
    cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
    cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

    def __init__(self, split_id=0):
        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams=2
        self.num_query_cams=2
        self.num_gallery_cams=2
        self.num_train_vids=num_train_tracklets
        self.num_query_vids=num_query_tracklets
        self.num_gallery_vids=num_gallery_tracklets

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.request
        url_opener.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = int(num_total_ids/2)

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
```

### 文件: `HierarchicalCrossAttention_model.py`

```python
import torch
import torch.nn as nn
from typing import Optional

from vit_ID import TransReID


class HierarchicalCrossAttentionReID(nn.Module):
    """
    Hierarchical cross-attention video matching model:
    - Shared ViT backbone (TransReID) extracts per-frame token sequences
    - Level 1 (spatial): frame-by-frame bidirectional cross attention over patch tokens (drop [CLS])
    - Level 2 (temporal): bidirectional cross attention over frame-level features along sequence dimension
    - Classifier head over concatenated video-level features -> single matching logit

    Input to forward: video1, video2 with shape [B, S, C, H, W]
    Output: logit with shape [B, 1]
    """
    
    def __init__(
        self,
        img_size=(256, 128),
        embed_dim: int = 768,
        num_heads: int = 12,
        stride_size: int = 16,
        camera_num: int = 6,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Shared backbone (classification head disabled)
        self.backbone = TransReID(
            img_size=img_size,
            patch_size=16,
            stride_size=stride_size,
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            camera=camera_num,
            drop_path_rate=0.0,
        )

        if pretrained_path is not None:
            try:
                self.backbone.load_param(pretrained_path)
            except Exception as exc:
                print(f"[HierarchicalCrossAttentionReID] Warning: failed to load pretrained weights: {exc}")

        # Level-1: spatial cross attention (per-frame, per-patch)
        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=False
        )

        # Level-2: temporal attention (over frame-level sequence)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=False
        )

        # Classifier head: concat([B,E],[B,E]) -> [B,1]
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim * 2, 1),
        )

    def _extract_patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, S, C, H, W]
        returns patch tokens (without [CLS]): [B, S, P, E]
        """
        bsz, seq_len, c, h, w = frames.shape
        frames = frames.view(bsz * seq_len, c, h, w)
        cam_labels = torch.zeros(bsz * seq_len, dtype=torch.long, device=frames.device)
        tokens = self.backbone(frames, cam_label=cam_labels)  # [B*S, N, E] (includes [CLS])
        patch_tokens = tokens[:, 1:, :]  # drop [CLS] -> [B*S, P, E]
        P = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(bsz, seq_len, P, -1)
        return patch_tokens

    def forward(self, video1: torch.Tensor, video2: torch.Tensor, return_rev_logit: bool = False) -> torch.Tensor:
        # a) feature extraction -> patch tokens [B, S, P, E]
        patch_tokens1 = self._extract_patch_tokens(video1)
        patch_tokens2 = self._extract_patch_tokens(video2)
        B, S, P, E = patch_tokens1.shape

        # b) Level-1: spatial cross attention (frame-by-frame)
        x1 = patch_tokens1.reshape(B * S, P, E)  # [B*S, P, E]
        x2 = patch_tokens2.reshape(B * S, P, E)  # [B*S, P, E]

        # i/ii) to (L, N, E) with L=P, N=B*S
        q1 = x1.permute(1, 0, 2)  # [P, B*S, E]
        k2v2 = x2.permute(1, 0, 2)  # [P, B*S, E]
        q2 = x2.permute(1, 0, 2)  # [P, B*S, E]
        k1v1 = x1.permute(1, 0, 2)  # [P, B*S, E]

        # iii) bidirectional spatial cross attention
        fused_spatial1, _ = self.spatial_cross_attention(query=q1, key=k2v2, value=k2v2)  # [P, B*S, E]
        fused_spatial2, _ = self.spatial_cross_attention(query=q2, key=k1v1, value=k1v1)  # [P, B*S, E]

        # iv) back to [B, S, P, E]
        fused_spatial1 = fused_spatial1.permute(1, 0, 2).contiguous().view(B, S, P, E)
        fused_spatial2 = fused_spatial2.permute(1, 0, 2).contiguous().view(B, S, P, E)

        # c) per-frame average pooling over patches -> [B, S, E]
        frame_feats1 = fused_spatial1.mean(dim=2)
        frame_feats2 = fused_spatial2.mean(dim=2)

        # d) Level-2: temporal attention over frame sequence
        t1 = frame_feats1.permute(1, 0, 2)  # [S, B, E]
        t2 = frame_feats2.permute(1, 0, 2)  # [S, B, E]
        fused_temporal1, _ = self.temporal_attention(query=t1, key=t2, value=t2)  # [S, B, E]
        fused_temporal2, _ = self.temporal_attention(query=t2, key=t1, value=t1)  # [S, B, E]
        fused_temporal1 = fused_temporal1.permute(1, 0, 2)  # [B, S, E]
        fused_temporal2 = fused_temporal2.permute(1, 0, 2)  # [B, S, E]

        # e) average pooling over time -> [B, E]
        video_feat1 = fused_temporal1.mean(dim=1)
        video_feat2 = fused_temporal2.mean(dim=1)

        # f/g) concat & classify -> [B, 1] or [B] 
        pair_feat = torch.cat([video_feat1, video_feat2], dim=-1)
        logit = self.classifier_head(pair_feat).squeeze(-1)  # [B]
        
        if not return_rev_logit:
            return logit
        
        # 反向顺序（video2, video1）使用同一个head，零额外参数
        pair_feat_rev = torch.cat([video_feat2, video_feat1], dim=-1)
        logit_rev = self.classifier_head(pair_feat_rev).squeeze(-1)  # [B]
        return logit, logit_rev
```

### 文件: `Loss_fun.py`

```python
import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss(num_classes):   
    
    feat_dim =768
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=3072, use_gpu=True) 
    
    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        

    def loss_func(score, feat, target, target_cam):
        if isinstance(score, list):
                ID_LOSS = [xent(scor, target) for scor in score[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                ID_LOSS = 0.25 * ID_LOSS + 0.75 * xent(score[0], target)
        else:
                ID_LOSS = xent(score, target)

        if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.25 * TRI_LOSS + 0.75 * triplet(feat[0], target)[0]

                center=center_criterion(feat[0], target)
                centr2 = [center_criterion2(feats, target) for feats in feat[1:]]
                centr2 = sum(centr2) / len(centr2)
                center=0.25 *centr2 +  0.75 *  center     
        else:
                TRI_LOSS = triplet(feat, target)[0]

        return   ID_LOSS+ TRI_LOSS, center

    return  loss_func,center_criterion
```

### 文件: `VID_Test.py`

```python
from Dataloader import dataloader
from VID_Trans_model import VID_Trans


from Loss_fun import make_loss
import random
import torch
import numpy as np
import os
import argparse


import logging
import os
import time
import torch
import torch.nn as nn

from torch.cuda import amp
from utility import AverageMeter, optimizer,scheduler

from torch.autograd import Variable   








def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
      for batch_idx, (imgs, pids, camids,_) in enumerate(queryloader):
       
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        
        b,  s, c, h, w = imgs.size()
        
        
        features = model(imgs,pids,cam_label=camids )
       
        features = features.view(b, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        
        q_pids.append(pids)
        q_camids.extend(camids)
      qf = torch.stack(qf)
      q_pids = np.asarray(q_pids)
      q_camids = np.asarray(q_camids)
      print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
      gf, g_pids, g_camids = [], [], []
      for batch_idx, (imgs, pids, camids,_) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, s,c, h, w = imgs.size()
        features = model(imgs,pids,cam_label=camids)
        features = features.view(b, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.append(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve r1:",cmc[0])
    
    return cmc[0], mAP



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VID-Trans-ReID")
    parser.add_argument(
        "--Dataset_name", default="", help="The name of the DataSet", type=str)
    parser.add_argument(
        "--model_path", default="", help="pretrained model", type=str)    
    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    pretrainpath=args.model_path

  

    train_loader,  num_query, num_classes, camera_num, view_num,q_val_set,g_val_set = dataloader(Dataset_name)
    model = VID_Trans( num_classes=num_classes, camera_num=camera_num,pretrainpath=None)

    device = "cuda"
    model=model.to(device)

    checkpoint = torch.load(pretrainpath)
    model.load_state_dict(checkpoint)

    
    model.eval()
    cmc,map = test(model, q_val_set,g_val_set)
    print('CMC: %.4f, mAP : %.4f'%(cmc,map))
```

### 文件: `VID_Trans_ReID.py`

```python
from Dataloader import dataloader
from VID_Trans_model import VID_Trans


from Loss_fun import make_loss

import random
import torch
import numpy as np
import os
import argparse

import logging
import os
import time
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torch.cuda import amp
import torch.distributed as dist

from utility import AverageMeter, optimizer,scheduler



   
        

       
from torch.autograd import Variable              
def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
      for batch_idx, (imgs, pids, camids,_) in enumerate(queryloader):
       
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        
        b,  s, c, h, w = imgs.size()
        
        
        features = model(imgs,pids,cam_label=camids )
       
        features = features.view(b, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        
        q_pids.append(pids)
        q_camids.extend(camids)
      qf = torch.stack(qf)
      q_pids = np.asarray(q_pids)
      q_camids = np.asarray(q_camids)
      print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
      gf, g_pids, g_camids = [], [], []
      for batch_idx, (imgs, pids, camids,_) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, s,c, h, w = imgs.size()
        features = model(imgs,pids,cam_label=camids)
        features = features.view(b, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.append(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve r1:",cmc[0])
    
    return cmc[0], mAP



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VID-Trans-ReID")
    parser.add_argument(
        "--Dataset_name", default="", help="The name of the DataSet", type=str)
    args = parser.parse_args()
    Dataset_name=args.Dataset_name
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    train_loader,  num_query, num_classes, camera_num, view_num,q_val_set,g_val_set = dataloader(Dataset_name)
    model = VID_Trans( num_classes=num_classes, camera_num=camera_num,pretrainpath=pretrainpath)
    
    loss_fun,center_criterion= make_loss( num_classes=num_classes)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr= 0.5)
    
    optimizer= optimizer( model)
    scheduler = scheduler(optimizer)
    scaler = amp.GradScaler()

    #Train
    device = "cuda"
    epochs = 120
    model=model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    cmc_rank1=0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        
        scheduler.step(epoch)
        model.train()
        
        for Epoch_n, (img, pid, target_cam,labels2) in enumerate(train_loader):
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            
            labels2=labels2.to(device)
            with amp.autocast(enabled=True):
                target_cam=target_cam.view(-1)
                score, feat ,a_vals= model(img, pid, cam_label=target_cam)
                
                labels2=labels2.to(device)
                attn_noise  = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()
                
                loss_id ,center= loss_fun(score, feat, pid, target_cam)
                loss=loss_id+ 0.0005*center +attn_loss
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            for param in center_criterion.parameters():
                    param.grad.data *= (1. / 0.0005)
            scaler.step(optimizer_center)
            scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (Epoch_n + 1) % 50 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (Epoch_n + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        if (epoch+1)%10 == 0 :
               
               model.eval()
               cmc,map = test(model, q_val_set,g_val_set)
               print('CMC: %.4f, mAP : %.4f'%(cmc,map))
               if cmc_rank1 < cmc:
                  cmc_rank1=cmc
                  torch.save(model.state_dict(),os.path.join('/VID-Trans-ReID',  Dataset_name+'Main_Model.pth'))
```

### 文件: `VID_Trans_model.py`

```python
import torch
import torch.nn as nn
import copy
from vit_ID import TransReID,Block
from functools import partial
from torch.nn import functional as F


def TCSS(features, shift, b,t):
    #aggregate features at patch level
    features=features.view(b,features.size(1),t*features.size(2))
    token = features[:, 0:1]

    batchsize = features.size(0)
    dim = features.size(-1)
    
    
    #shift the patches with amount=shift
    features= torch.cat([features[:, shift:], features[:, 1:shift]], dim=1)
    
    # Patch Shuffling by 2 part
    try:
        features = features.view(batchsize, 2, -1, dim)
    except:
        features = torch.cat([features, features[:, -2:-1, :]], dim=1)
        features = features.view(batchsize, 2, -1, dim)
    
    features = torch.transpose(features, 1, 2).contiguous()
    features = features.view(batchsize, -1, dim)
    
    return features,token    

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)




class VID_Trans(nn.Module):
    def __init__(self, num_classes, camera_num,pretrainpath):
        super(VID_Trans, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes
        
        
        self.base =TransReID(
        img_size=[256, 128], patch_size=16, stride_size=[16, 16], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera_num,  drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,norm_layer=partial(nn.LayerNorm, eps=1e-6),  cam_lambda=3.0)
        
          
        state_dict = torch.load(pretrainpath, map_location='cpu')
        self.base.load_param(state_dict,load=True)
        
       
        #global stream
        block= self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
       
        #-----------------------------------------------
        #-----------------------------------------------
 

        # building local video stream
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]  # stochastic depth decay rule
        
        self.block1 = Block(
                dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0, attn_drop=0, drop_path=dpr[11], norm_layer=partial(nn.LayerNorm, eps=1e-6))
       
        self.b2 = nn.Sequential(
            self.block1,
            nn.LayerNorm(3072)#copy.deepcopy(layer_norm)
        )
        
        
        self.bottleneck_1 = nn.BatchNorm1d(3072)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(3072)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(3072)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(3072)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)


        self.classifier_1 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(3072, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)


        #-------------------video attention-------------
        self.middle_dim = 256 # middle layer dimension
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1,1]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 
        #------------------------------------------
        
        self.shift_num = 5
        self.part = 4
        self.rearrange=True 
        



    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        b=x.size(0)
        t=x.size(1)
        
        x=x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        features = self.base(x, cam_label=cam_label)
        
        
        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]
        
        global_feat=global_feat.unsqueeze(dim=2)
        global_feat=global_feat.unsqueeze(dim=3)
        a = F.relu(self.attention_conv(global_feat))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a_vals = a 
        
        a = F.softmax(a, dim=1)
        x = global_feat.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        global_feat = att_x.view(b,self.in_planes)
        feat = self.bottleneck(global_feat)
        



        #-------------------------------------------------
        #-------------------------------------------------


        # video patch patr features

        feature_length = features.size(1) - 1
        patch_length = feature_length // 4
        
        #Temporal clip shift and shuffled
        x ,token=TCSS(features, self.shift_num, b,t)  
        
           
        # part1
        part1 = x[:, :patch_length]
        part1 = self.b2(torch.cat((token, part1), dim=1))
        part1_f = part1[:, 0]

        # part2
        part2 = x[:, patch_length:patch_length*2]
        part2 = self.b2(torch.cat((token, part2), dim=1))
        part2_f = part2[:, 0]

        # part3
        part3 = x[:, patch_length*2:patch_length*3]
        part3 = self.b2(torch.cat((token, part3), dim=1))
        part3_f = part3[:, 0]

        # part4
        part4 = x[:, patch_length*3:patch_length*4]
        part4 = self.b2(torch.cat((token, part4), dim=1))
        part4_f = part4[:, 0]
       
        
        
        part1_bn = self.bottleneck_1(part1_f)
        part2_bn = self.bottleneck_2(part2_f)
        part3_bn = self.bottleneck_3(part3_f)
        part4_bn = self.bottleneck_4(part4_f)
        
        if self.training:
            
            Global_ID = self.classifier(feat)
            Local_ID1 = self.classifier_1(part1_bn)
            Local_ID2 = self.classifier_2(part2_bn)
            Local_ID3 = self.classifier_3(part3_bn)
            Local_ID4 = self.classifier_4(part4_bn)
                
            return [Global_ID, Local_ID1, Local_ID2, Local_ID3, Local_ID4 ], [global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals #[global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals 
        
        else:
              return torch.cat([feat, part1_bn/4 , part2_bn/4 , part3_bn /4, part4_bn/4 ], dim=1)
            


    def load_param(self, trained_path,load=False):
        if not load:
            param_dict = torch.load(trained_path)
            for i in param_dict:
               self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
               print('Loading pretrained model from {}'.format(trained_path))
        else:
            param_dict=trained_path
            for i in param_dict:
             #print(i)   
             if i not in self.state_dict() or 'classifier' in i or 'sie_embed' in i:
                continue
             self.state_dict()[i].copy_(param_dict[i])
           
            
            
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
```

### 文件: `loss/center_loss.py`

```python
from __future__ import absolute_import

import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


if __name__ == '__main__':
    use_gpu = False
    center_loss = CenterLoss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    print(loss)
```

### 文件: `loss/softmax_loss.py`

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

### 文件: `loss/triplet_loss.py`

```python
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
```

### 文件: `pair_losses.py`

```python
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
```

### 文件: `rerank_evaluate.py`

```python
import argparse
import os
import time
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from Dataloader import dataloader, read_image
from VID_Trans_model import VID_Trans
from HierarchicalCrossAttention_model import HierarchicalCrossAttentionReID


# -----------------------------
# Metrics (copied to avoid import coupling)
# -----------------------------

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


# -----------------------------
# Utilities
# -----------------------------

def make_val_transform():
    return T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def sample_center_clip(img_paths, seq_len: int, transform) -> torch.Tensor:
    """Deterministic center sampling of a clip with length seq_len.
    Returns a tensor [S, C, H, W].
    """
    num = len(img_paths)
    frame_indices = list(range(num))
    if num <= seq_len:
        indices = list(frame_indices)
        indices.extend([indices[-1] for _ in range(seq_len - len(indices))])
    else:
        begin_index = (num - seq_len) // 2
        end_index = begin_index + seq_len
        indices = frame_indices[begin_index:end_index]
    imgs = []
    for idx in indices:
        img_path = img_paths[int(idx)]
        img = read_image(img_path)
        if transform is not None:
            img = transform(img)
        imgs.append(img.unsqueeze(0))
    return torch.cat(imgs, dim=0)  # [S, C, H, W]


@torch.no_grad()
def extract_vidtrans_features(model: nn.Module, dataset, device: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Replicates logic from VID_Test.test() to extract features per tracklet.
    dataset items: returns (imgs_array [B,S,C,H,W], pid, camids_list, img_paths)
    Returns:
      feats: torch.FloatTensor [num_tracks, D]
      pids: np.ndarray [num_tracks]
      camids: np.ndarray [num_tracks] (one camera id per tracklet for evaluation)
    """
    model.eval()
    feats, pids_arr, camids_arr = [], [], []
    for (imgs, pid, camids, _img_paths) in dataset:
        # imgs: [B, S, C, H, W]
        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(device)
        b, s, c, h, w = imgs.size()
        # Use per-frame cam ids for feature extraction (matches flattened frames)
        feat = model(imgs, pid, cam_label=camids)  # [B, D]
        feat = feat.view(b, -1)
        feat = torch.mean(feat, dim=0)  # [D]
        feats.append(feat.cpu())
        pids_arr.append(pid)
        # For evaluation, keep ONE camid per tracklet (all frames in a tracklet share the same cam)
        if isinstance(camids, (list, tuple, np.ndarray)) and len(camids) > 0:
            camids_arr.append(int(camids[0]))
        else:
            camids_arr.append(int(camids))
    feats = torch.stack(feats, dim=0)
    pids_np = np.asarray(pids_arr)
    camids_np = np.asarray(camids_arr)
    return feats, pids_np, camids_np


def compute_euclidean_dist(qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) \
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    return distmat.cpu().numpy()


def safe_torch_load(path: str, map_location="cpu"):
    """Load checkpoints safely across torch versions.
    Tries weights_only=False first (PyTorch>=2.6), falls back if unsupported.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg
        return torch.load(path, map_location=map_location)


def infer_camera_num_from_state_dict(st) -> int:
    """Infer camera_num from checkpoint state_dict by inspecting Cam embedding shape.
    Returns None if not found.
    """
    if not isinstance(st, dict):
        return None
    # common keys: saved either as Parameter (no .weight) or as weight tensor
    for k in ("backbone.Cam", "module.backbone.Cam", "backbone.Cam.weight", "module.backbone.Cam.weight"):
        if k in st:
            v = st[k]
            shp = getattr(v, "shape", None)
            if shp and len(shp) > 0:
                return int(shp[0])
    # fallback: search any *Cam or *Cam.weight
    for k, v in st.items():
        if k.endswith("Cam") or k.endswith("Cam.weight"):
            shp = getattr(v, "shape", None)
            if shp and len(shp) > 0:
                return int(shp[0])
    return None


@torch.no_grad()
def rerank_with_hierarchical(
    model: nn.Module,
    q_set,
    g_set,
    distmat: np.ndarray,
    topk: int,
    seq_len: int,
    device: str,
    batch_size: int = 16,
    score_mode: str = "1-minus-prob",
    fuse_mode: str = "replace",
    fuse_lambda: float = 0.2,
    logit_temp: float = 1.0,
    gate_mode: str = "prob",
    gate_tau: float = 0.7,
    bidirectional: bool = False,
    norm_hier: str = "minmax",
    debug_stats: bool = False,
    corr_gate_thr: float = 0.0,
    corr_power: float = 1.0,
    topk_frac: float = 1.0,
    rank_weighting: str = "none",
    keep_top1: bool = False,
    base_gap_thr: float = 0.0,
    base_gap_power: float = 1.0,
    keep_top1_if_confident: bool = False,
) -> np.ndarray:
    """For each query, refine top-K gallery distances using the hierarchical re-ranker.
    Replaces top-K baseline distances by a mapping of sigmoid(logit) from the re-ranker.
    If score_mode == '1-minus-prob', distance = 1 - prob (prob treated as similarity, default).
    If score_mode == 'prob', distance = prob (useful if model outputs are inverted on this dataset).
    """
    model.eval()
    transform = make_val_transform()
    num_q, num_g = distmat.shape
    dist_rerank = distmat.copy()

    # stats
    stats_total_pairs = 0
    stats_modified_pairs = 0
    stats_delta_norm_sum = 0.0
    stats_top1_changed = 0
    stats_corr_list = []

    for qi in range(num_q):
        # top-K selection
        order = np.argsort(distmat[qi])
        cand = order[: min(topk, num_g)]

        if len(cand) == 0:
            continue

        # baseline distances for full top-K and its normalized range
        base_top_full = distmat[qi, cand].astype(np.float32)
        bmin = float(base_top_full.min())
        bmax = float(base_top_full.max())
        if bmax > bmin:
            base_top_norm = (base_top_full - bmin) / (bmax - bmin)
        else:
            base_top_norm = np.zeros_like(base_top_full)

        # build query clip once
        q_img_paths, _q_pid, _q_camid = q_set.dataset[qi]
        vq = sample_center_clip(q_img_paths, seq_len=seq_len, transform=transform)  # [S,C,H,W]

        # build gallery clips for top-K
        vg_list = []
        for gj in cand:
            g_img_paths, _g_pid, _g_camid = g_set.dataset[gj]
            vg = sample_center_clip(g_img_paths, seq_len=seq_len, transform=transform)
            vg_list.append(vg)

        if len(vg_list) == 0:
            continue

        vg_tensor = torch.stack(vg_list, dim=0)  # [K,S,C,H,W]
        vq_tensor = vq.unsqueeze(0).expand(vg_tensor.size(0), -1, -1, -1, -1)  # [K,S,C,H,W]

        # collect probabilities for entire top-K
        probs_all = np.zeros((vg_tensor.size(0),), dtype=np.float32)

        start = 0
        while start < vg_tensor.size(0):
            end = min(start + batch_size, vg_tensor.size(0))
            v1 = vq_tensor[start:end].to(device)
            v2 = vg_tensor[start:end].to(device)
            logits = model(v1, v2)  # [B,1]
            p1 = torch.sigmoid(logits / max(1e-6, logit_temp)).view(-1)
            if bidirectional:
                logits_rev = model(v2, v1)
                p2 = torch.sigmoid(logits_rev / max(1e-6, logit_temp)).view(-1)
                p = (p1 + p2) * 0.5
            else:
                p = p1
            probs_all[start:end] = p.detach().cpu().numpy()
            start = end

        # optional normalization of hierarchical scores across top-K
        probs_use = probs_all.copy()
        if norm_hier == "minmax":
            pmin = float(probs_all.min())
            pmax = float(probs_all.max())
            if pmax > pmin:
                probs_use = (probs_all - pmin) / (pmax - pmin)
            else:
                probs_use = np.zeros_like(probs_all) + 0.5
        elif norm_hier == "zscore":
            mu = float(probs_all.mean())
            sigma = float(probs_all.std())
            z = (probs_all - mu) / max(sigma, 1e-6)
            probs_use = 1.0 / (1.0 + np.exp(-z))

        # map to distance form in [0,1] if needed
        score_dist = (1.0 - probs_use) if (score_mode == "1-minus-prob") else probs_use

        # ----- query-adaptive settings -----
        # correlation between hierarchical prob (pre-norm) and baseline similarity (higher is better)
        s_base_for_corr = 1.0 - base_top_norm
        if np.std(probs_all) > 1e-8 and np.std(s_base_for_corr) > 1e-8:
            try:
                corr_q = float(np.corrcoef(probs_all, s_base_for_corr)[0, 1])
            except Exception:
                corr_q = 0.0
        else:
            corr_q = 0.0

        # correlation-gated effective lambda
        if corr_q <= corr_gate_thr:
            lambda_eff = 0.0
        else:
            scale = (corr_q - corr_gate_thr) / max(1e-6, (1.0 - corr_gate_thr))
            lambda_eff = float(fuse_lambda) * (scale ** float(corr_power))

        # rank-based weighting vector within top-K
        K = len(cand)
        if rank_weighting == "downweight_top":
            # 0 for best baseline, ->1 for worst in top-K
            if K > 1:
                weights = np.linspace(0.0, 1.0, K, dtype=np.float32)
            else:
                weights = np.array([1.0], dtype=np.float32)
        elif rank_weighting == "upweight_top":
            if K > 1:
                weights = np.linspace(1.0, 0.0, K, dtype=np.float32)
            else:
                weights = np.array([1.0], dtype=np.float32)
        else:
            weights = np.ones((K,), dtype=np.float32)

        lam_vec = lambda_eff * weights

        # query-level baseline uncertainty gating based on top-1 vs top-2 normalized gap
        if base_gap_thr > 0.0 and K > 1:
            gap_norm = float(base_top_norm[1] - base_top_norm[0])  # smaller gap => more uncertain baseline
            if gap_norm >= base_gap_thr:
                # baseline confident -> disable fusion
                lam_vec = lam_vec * 0.0
            else:
                # scale by uncertainty strength
                uncert_scale = ((base_gap_thr - gap_norm) / max(1e-6, base_gap_thr)) ** float(base_gap_power)
                lam_vec = lam_vec * float(uncert_scale)

        # top-fraction to modify (by confidence |p-0.5|)
        frac = float(topk_frac)
        if frac >= 1.0:
            modify_mask_frac = np.ones((K,), dtype=bool)
        elif frac <= 0.0:
            modify_mask_frac = np.zeros((K,), dtype=bool)
        else:
            m = max(1, int(math.ceil(frac * K)))
            conf = np.abs(probs_all - 0.5)
            idx_sorted = np.argsort(-conf)  # high confidence first
            modify_idx = idx_sorted[:m]
            modify_mask_frac = np.zeros((K,), dtype=bool)
            modify_mask_frac[modify_idx] = True

        # fuse on normalized scale, then map back to [bmin, bmax]
        if fuse_mode == "replace":
            new_norm = score_dist.copy()
            if gate_mode == "prob":
                for i in range(len(cand)):
                    p = probs_all[i]
                    if not ((p >= gate_tau) or (p <= 1.0 - gate_tau)):
                        new_norm[i] = base_top_norm[i]
            # also apply topk_frac gating
            for i in range(len(cand)):
                if not modify_mask_frac[i]:
                    new_norm[i] = base_top_norm[i]
            new_norm = np.clip(new_norm, 0.0, 1.0)
            new_vals = new_norm * (bmax - bmin) + bmin
        elif fuse_mode == "add":
            # element-wise lambda
            new_norm = base_top_norm + lam_vec * score_dist
            if gate_mode == "prob":
                for i in range(len(cand)):
                    p = probs_all[i]
                    if not ((p >= gate_tau) or (p <= 1.0 - gate_tau)):
                        new_norm[i] = base_top_norm[i]
            # topk_frac gating
            for i in range(len(cand)):
                if not modify_mask_frac[i]:
                    new_norm[i] = base_top_norm[i]
            new_norm = np.clip(new_norm, 0.0, 1.0)
            new_vals = new_norm * (bmax - bmin) + bmin
        else:  # add_sim
            s_base = 1.0 - base_top_norm
            s_hier = probs_use
            # element-wise convex combination using lam_vec
            fused_s = (1.0 - lam_vec) * s_base + lam_vec * s_hier
            d_fused_norm = 1.0 - fused_s
            if gate_mode == "prob":
                for i in range(len(cand)):
                    p = probs_all[i]
                    if not ((p >= gate_tau) or (p <= 1.0 - gate_tau)):
                        d_fused_norm[i] = base_top_norm[i]
            # topk_frac gating
            for i in range(len(cand)):
                if not modify_mask_frac[i]:
                    d_fused_norm[i] = base_top_norm[i]
            d_fused_norm = np.clip(d_fused_norm, 0.0, 1.0)
            new_vals = d_fused_norm * (bmax - bmin) + bmin

        # keep baseline top-1 within top-K unchanged if requested (unconditional),
        # or conditionally when baseline is confident (gap >= base_gap_thr)
        do_keep_top1 = bool(keep_top1)
        if (not do_keep_top1) and keep_top1_if_confident and (base_gap_thr > 0.0) and (K > 1):
            gap_norm2 = float(base_top_norm[1] - base_top_norm[0])
            if gap_norm2 >= base_gap_thr:
                do_keep_top1 = True
        if do_keep_top1 and len(cand) > 0:
            new_vals[0] = base_top_full[0]

        if debug_stats:
            
            # modification mask in normalized space
            if 'new_norm' in locals():
                diff_norm = np.abs(new_norm - base_top_norm)
            else:
                diff_norm = np.abs(d_fused_norm - base_top_norm)
            mod_mask = diff_norm > 1e-6
            stats_modified_pairs += int(mod_mask.sum())
            stats_total_pairs += len(cand)
            stats_delta_norm_sum += float(diff_norm.sum())
            # correlation between hierarchical prob and baseline similarity
            s_base_corr = 1.0 - base_top_norm
            if np.std(probs_all) > 1e-8 and np.std(s_base_corr) > 1e-8:
                try:
                    corr = float(np.corrcoef(probs_all, s_base_corr)[0, 1])
                    if not np.isnan(corr):
                        stats_corr_list.append(corr)
                except Exception:
                    pass
            # top-1 change within top-K
            new_top_idx = int(np.argmin(new_vals))
            if new_top_idx != 0:
                stats_top1_changed += 1

        # write back all top-K at once
        for idx_local, gj in enumerate(cand):
            dist_rerank[qi, gj] = new_vals[idx_local]

        if (qi + 1) % 50 == 0:
            print(f"Re-ranked queries: {qi + 1}/{num_q}")

    if debug_stats and stats_total_pairs > 0:
        mod_rate = stats_modified_pairs / float(stats_total_pairs)
        mean_delta_norm = stats_delta_norm_sum / float(stats_total_pairs)
        mean_corr = (sum(stats_corr_list) / len(stats_corr_list)) if len(stats_corr_list) > 0 else float('nan')
        print("[STATS] pairs modified: {} / {} ({:.1%})".format(stats_modified_pairs, stats_total_pairs, mod_rate))
        print("[STATS] mean |Δ| in normalized distance across all top-K pairs: {:.6f}".format(mean_delta_norm))
        print("[STATS] queries with top-1 (within top-K) changed: {} / {}".format(stats_top1_changed, num_q))
        print("[STATS] mean corr(prob, baseline_similarity) over queries: {:.4f}".format(mean_corr))

    return dist_rerank


def parse_args():
    parser = argparse.ArgumentParser(description="Two-Stage Re-Ranking Evaluation")
    parser.add_argument("--Dataset_name", type=str, default="Mars")
    parser.add_argument("--vid_pretrain", type=str, default="jx_vit_base_p16_224-80ecf9dd.pth",
                        help="ViT backbone pretrained weights for VID_Trans and backbone in re-ranker")
    parser.add_argument("--baseline_ckpt", type=str, default=os.path.join("checkpoints_baseline", "baseline_best_auc_0.9863.pth"))
    parser.add_argument("--hierarchical_ckpt", type=str, default=os.path.join("checkpoints_hierarchical", "hierarchical_best_auc_0.9863.pth"))
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for re-ranker pairs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score_mode", type=str, default="1-minus-prob", choices=["1-minus-prob", "prob"],
                        help="How to map sigmoid(logit) to distance for re-ranking. Default: 1-minus-prob (treat prob as similarity). Use 'prob' if scores seem inverted.")
    parser.add_argument("--fuse_mode", type=str, default="replace", choices=["replace", "add", "add_sim"],
                        help="How to apply hierarchical score: replace top-K distances; add on normalized distance; or add_sim on similarity (safer).")
    parser.add_argument("--fuse_lambda", type=float, default=0.2,
                        help="Weight for hierarchical score in additive fusion (effective when --fuse_mode add).")
    parser.add_argument("--logit_temp", type=float, default=1.0,
                        help="Temperature for sigmoid(logit/T) to calibrate scores.")
    parser.add_argument("--gate_mode", type=str, default="prob", choices=["none", "prob"],
                        help="Gating strategy: 'prob' only modifies pairs when model is confident (p>=tau or p<=1-tau).")
    parser.add_argument("--gate_tau", type=float, default=0.7,
                        help="Confidence threshold for gating when --gate_mode prob.")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Average scores of model(q,g) and model(g,q) to reduce variance.")
    parser.add_argument("--norm_hier", type=str, default="minmax", choices=["none", "minmax", "zscore"],
                        help="Normalization for hierarchical scores within each query's top-K.")
    parser.add_argument("--debug_stats", action="store_true",
                        help="Print modification stats (fraction changed, avg delta, correlation) for diagnostics.")
    parser.add_argument("--corr_gate_thr", type=float, default=0.0,
                        help="Query-level correlation threshold with baseline similarity to enable fusion (0~1). Below this, lambda shrinks to 0.")
    parser.add_argument("--corr_power", type=float, default=1.0,
                        help="Exponent for correlation-based scaling of lambda (>=1 for conservative).")
    parser.add_argument("--topk_frac", type=float, default=1.0,
                        help="Fraction (0~1] of top-K pairs to actually modify ranked by |p-0.5| confidence.")
    parser.add_argument("--rank_weighting", type=str, default="none", choices=["none", "downweight_top", "upweight_top"],
                        help="Rank-based weighting of fusion strength within top-K.")
    parser.add_argument("--keep_top1", action="store_true",
                        help="Never change baseline top-1 candidate within top-K (safeguard Rank-1).")
    parser.add_argument("--base_gap_thr", type=float, default=0.0,
                        help="Enable fusion only when baseline top-1 vs top-2 normalized gap is below this threshold (baseline uncertain). 0 disables.")
    parser.add_argument("--base_gap_power", type=float, default=1.0,
                        help="Exponent for scaling lambda by baseline uncertainty: ((thr-gap)/thr)^power.")
    parser.add_argument("--keep_top1_if_confident", action="store_true",
                        help="Only keep baseline top-1 when baseline is confident (gap >= base_gap_thr). Allows improving Rank-1 on uncertain queries.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Load datasets and meta
    train_loader, num_query, num_classes, camera_num, view_num, q_set, g_set = dataloader(args.Dataset_name)

    # Stage-1: Baseline feature extractor
    vid_model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.vid_pretrain).to(device)
    # Load baseline checkpoint (state_dict) if compatible; fall back to only ImageNet pretrain otherwise
    try:
        print(f"Loading baseline checkpoint from: {args.baseline_ckpt}")
        base_state = safe_torch_load(args.baseline_ckpt, map_location="cpu")
        if isinstance(base_state, dict) and "model" in base_state:
            base_state = base_state["model"]
        vid_model.load_state_dict(base_state, strict=False)
    except Exception as exc:
        print(f"[WARN] Could not load baseline checkpoint into VID_Trans: {exc}. Proceeding with backbone pretrain only.")
    vid_model.eval()

    # Extract features
    print("Extracting baseline features (query)...")
    qf, q_pids, q_camids = extract_vidtrans_features(vid_model, q_set, device)
    print("Extracting baseline features (gallery)...")
    gf, g_pids, g_camids = extract_vidtrans_features(vid_model, g_set, device)

    # Compute baseline distance matrix and metrics
    print("Computing baseline distances and metrics...")
    dist_base = compute_euclidean_dist(qf, gf)
    cmc_base, mAP_base = evaluate(dist_base, q_pids, g_pids, q_camids, g_camids)
    print("Baseline Results ----------")
    print(f"mAP: {mAP_base:.1%}")
    print(f"Rank-1: {cmc_base[0]:.1%}")

    # Stage-2: Hierarchical re-ranker
    print(f"Loading re-ranker checkpoint from: {args.hierarchical_ckpt}")
    hier_obj = safe_torch_load(args.hierarchical_ckpt, map_location="cpu")
    # Resolve to plain state_dict (support {'model': ...} or {'state_dict': ...})
    if isinstance(hier_obj, dict) and "model" in hier_obj:
        state = hier_obj["model"]
    elif isinstance(hier_obj, dict) and "state_dict" in hier_obj:
        state = hier_obj["state_dict"]
    else:
        state = hier_obj

    # Infer camera_num from checkpoint if possible
    ckpt_cam_num = infer_camera_num_from_state_dict(state)
    use_cam_num = ckpt_cam_num if ckpt_cam_num is not None else camera_num
    print(f"[INFO] Dataset camera_num={camera_num}; Checkpoint camera_num={ckpt_cam_num}; Using camera_num={use_cam_num}")

    print("Initializing Hierarchical Cross-Attention re-ranker...")
    rerank_model = HierarchicalCrossAttentionReID(
        img_size=(256, 128),
        embed_dim=768,
        num_heads=12,
        stride_size=16,
        camera_num=use_cam_num,
        pretrained_path=args.vid_pretrain,
    ).to(device)

    model_state = rerank_model.state_dict()

    def _filter_and_strip_prefix(st):
        # 1) try direct key match with same shapes
        filtered = {k: v for k, v in st.items() if k in model_state and getattr(v, 'shape', None) == model_state[k].shape}
        if len(filtered) == 0:
            # 2) try removing 'module.' prefix (from DataParallel)
            stripped = { (k[7:] if k.startswith('module.') else k): v for k, v in st.items() }
            filtered2 = {k: v for k, v in stripped.items() if k in model_state and getattr(v, 'shape', None) == model_state[k].shape}
            return filtered2, stripped
        return filtered, st

    filtered_state, used_source = _filter_and_strip_prefix(state)
    dropped = [k for k in used_source.keys() if k not in filtered_state]
    if dropped:
        # Show only a few dropped keys for brevity (likely includes 'backbone.Cam')
        print(f"[INFO] Skipped {len(dropped)} keys due to mismatch, e.g., {dropped[:3]}")
    rerank_model.load_state_dict(filtered_state, strict=False)
    rerank_model.eval()

    # Re-rank top-K per query
    print(f"Re-ranking top-{args.topk} per query...")
    start = time.time()
    dist_rerank = rerank_with_hierarchical(
        model=rerank_model,
        q_set=q_set,
        g_set=g_set,
        distmat=dist_base,
        topk=args.topk,
        seq_len=args.seq_len,
        device=device,
        batch_size=args.batch_size,
        score_mode=args.score_mode,
        fuse_mode=args.fuse_mode,
        fuse_lambda=args.fuse_lambda,
        logit_temp=args.logit_temp,
        gate_mode=args.gate_mode,
        gate_tau=args.gate_tau,
        bidirectional=args.bidirectional,
        norm_hier=args.norm_hier,
        debug_stats=args.debug_stats,
        corr_gate_thr=args.corr_gate_thr,
        corr_power=args.corr_power,
        topk_frac=args.topk_frac,
        rank_weighting=args.rank_weighting,
        keep_top1=args.keep_top1,
        base_gap_thr=args.base_gap_thr,
        base_gap_power=args.base_gap_power,
        keep_top1_if_confident=args.keep_top1_if_confident,
    )
    elapsed = time.time() - start
    print(f"Re-ranking done in {elapsed:.1f}s")

    # Evaluate re-ranked matrix
    cmc_r, mAP_r = evaluate(dist_rerank, q_pids, g_pids, q_camids, g_camids)
    print("Re-ranked Results ----------")
    print(f"mAP: {mAP_r:.1%}")
    print(f"Rank-1: {cmc_r[0]:.1%}")


if __name__ == "__main__":
    main()
```

### 文件: `train_baseline.py`

```python
import argparse
import os
import time

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
    return parser.parse_args()


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
```

### 文件: `train_hierarchical.py`

```python
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
```

### 文件: `utility.py`

```python
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import torch
import logging
import math
import torch
from typing import Dict, Any
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RandomErasing3(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img , 0 
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img , 1
            return img , 0         
        

def scheduler(optimizer):
    num_epochs = 120
    
    lr_min = 0.002 * 0.008
    warmup_lr_init = 0.01 * 0.008
    
    warmup_t = 5
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler        




def optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.008
        weight_decay = 1e-4
        if "bias" in key:
            lr = 0.008 * 2
            weight_decay = 1e-4

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    
    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)

    
    
    return optimizer





class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs

class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))
```

### 文件: `vit_ID.py`

```python
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable as _Iterable


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, _Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
       
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x





class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, 
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, cam_lambda =3.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
       
        self.cam_num = camera
        self.cam_lambda = cam_lambda


        self.patch_embed = PatchEmbed_overlap(img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.Cam = nn.Parameter(torch.zeros(camera, 1, embed_dim))

        trunc_normal_(self.Cam, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id):
        B = x.shape[0]
       
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed + self.cam_lambda * self.Cam[camera_id]
        x = self.pos_drop(x)

        for blk in self.blocks[:-1]:
                x = blk(x)
        return x

        
    def forward(self, x, cam_label=None):
        x = self.forward_features(x, cam_label)
        return x

    def load_param(self, model_path,load=False):
        if not load:
            param_dict = torch.load(model_path, map_location='cpu')
        else:
            param_dict=  model_path  
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb





def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
```
