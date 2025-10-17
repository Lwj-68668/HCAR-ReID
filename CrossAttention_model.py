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


