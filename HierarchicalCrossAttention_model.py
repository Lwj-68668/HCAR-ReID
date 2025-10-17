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
