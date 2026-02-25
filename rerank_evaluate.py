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
def extract_vidtrans_features(model: nn.Module, dataset, device: str, verbose: bool = False, legacy_camid: bool = False) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Replicates logic from VID_Test.test() to extract features per tracklet with dense sampling.
    dataset items: returns (imgs_array [B,S,C,H,W], pid, camids_list, img_paths)
    where B is the number of clips from dense sampling (typically 10-50 clips per video)
    
    Args:
        legacy_camid: If True, use original VID_Test.py camid collection (extend all frame camids).
                      This replicates the original evaluation behavior to match checkpoint metrics.
                      If False, use correct one-camid-per-tracklet (recommended for accurate evaluation).
    Returns:
      feats: torch.FloatTensor [num_tracks, D]
      pids: np.ndarray [num_tracks]
      camids: np.ndarray - shape depends on legacy_camid flag
    """
    model.eval()
    feats, pids_arr, camids_arr = [], [], []
    total_clips = 0
    clip_counts = []
    
    for idx, (imgs, pid, camids, _img_paths) in enumerate(dataset):
        # imgs: [B, S, C, H, W] where B = num_clips from dense sampling
        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(device)
        b, s, c, h, w = imgs.size()
        
        # 调试信息：记录clips数量
        clip_counts.append(b)
        total_clips += b
        
        if verbose and idx < 3:
            print(f"  [Sample {idx}] Video has {b} clips, each with {s} frames, shape: [{b},{s},{c},{h},{w}]")
        
        # Use per-frame cam ids for feature extraction (matches flattened frames)
        feat = model(imgs, pid, cam_label=camids)  # [B, D]
        feat = feat.view(b, -1)  # [B, D]
        feat = torch.mean(feat, dim=0)  # 对B个clips求平均 → [D]
        feats.append(feat.cpu())
        pids_arr.append(pid)
        
        # Camid collection: legacy vs correct
        if legacy_camid:
            # Original VID_Test.py behavior: extend all frame camids
            # This creates length mismatch with pids but replicates original evaluation
            if isinstance(camids, (list, tuple, np.ndarray)):
                camids_arr.extend([int(c) for c in camids])
            else:
                camids_arr.append(int(camids))
        else:
            # Correct behavior: one camid per tracklet
            if isinstance(camids, (list, tuple, np.ndarray)) and len(camids) > 0:
                camids_arr.append(int(camids[0]))
            else:
                camids_arr.append(int(camids))
    
    feats = torch.stack(feats, dim=0)
    pids_np = np.asarray(pids_arr)
    camids_np = np.asarray(camids_arr)
    
    # 打印统计信息
    if verbose:
        avg_clips = total_clips / len(dataset) if len(dataset) > 0 else 0
        print(f"  ✓ Total videos: {len(dataset)}, Total clips: {total_clips}, Avg clips/video: {avg_clips:.1f}")
        print(f"  ✓ Clips range: min={min(clip_counts)}, max={max(clip_counts)}")
        if legacy_camid:
            print(f"  ⚠️  Legacy camid mode: camids array has {len(camids_np)} elements (vs {len(pids_np)} pids)")
        else:
            print(f"  ✓ Correct camid mode: camids array has {len(camids_np)} elements (matching {len(pids_np)} pids)")
    
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
    if isinstance(path, str) and os.sep == "/" and "\\" in path:
        path = path.replace("\\", "/")
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

    # Debug: collect all probs for analysis
    all_probs_debug = []
    
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
        
        all_probs_debug.extend(probs_all.tolist())

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
        # Debug: probability distribution analysis
        if all_probs_debug:
            probs_arr = np.array(all_probs_debug)
            print("[DEBUG] Prob distribution: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
                probs_arr.min(), probs_arr.max(), probs_arr.mean(), probs_arr.std()))

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
    parser.add_argument("--legacy_eval", action="store_true",
                        help="Use original VID_Test.py camid collection (extend all frame camids). "
                             "This replicates the original evaluation behavior to match checkpoint metrics. "
                             "Without this flag, uses correct one-camid-per-tracklet for accurate evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Load datasets and meta
    train_loader, num_query, num_classes, camera_num, view_num, q_set, g_set = dataloader(args.Dataset_name)

    # Stage-1: Baseline feature extractor
    # IMPORTANT: Set pretrainpath=None to avoid loading ImageNet weights that conflict with MARS checkpoint
    vid_model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=None).to(device)
    # Load baseline checkpoint (state_dict) if compatible; fall back to only ImageNet pretrain otherwise
    print("\n" + "="*80)
    print("STAGE 1: Loading Baseline VID-Trans-ReID Model")
    print("="*80)
    try:
        print(f"Loading baseline checkpoint from: {args.baseline_ckpt}")
        base_state = safe_torch_load(args.baseline_ckpt, map_location="cpu")
        
        # 显示checkpoint信息
        if isinstance(base_state, dict):
            print(f"\n[Checkpoint Info]")
            if "epoch" in base_state:
                print(f"  Epoch: {base_state['epoch']}")
            if "rank1" in base_state:
                print(f"  Original Rank-1: {base_state['rank1']:.4f} ({base_state['rank1']*100:.2f}%)")
            if "mAP" in base_state:
                print(f"  Original mAP: {base_state['mAP']:.4f} ({base_state['mAP']*100:.2f}%)")
            if "timestamp" in base_state:
                print(f"  Timestamp: {base_state['timestamp']}")
        
        # Handle different checkpoint formats: 'model', 'state_dict', or direct dict
        if isinstance(base_state, dict):
            if "model" in base_state:
                base_state = base_state["model"]
            elif "state_dict" in base_state:
                base_state = base_state["state_dict"]

        if isinstance(base_state, dict):
            base_state = {(k[7:] if k.startswith("module.") else k): v for k, v in base_state.items()}
        
        print(f"\n[Loading Parameters]")
        missing_keys, unexpected_keys = vid_model.load_state_dict(base_state, strict=False)
        print(f"  ✓ Loaded {len(base_state)} parameters from baseline checkpoint")
        if missing_keys:
            print(f"  ⚠️  Missing {len(missing_keys)} keys (will use random init): {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected {len(unexpected_keys)} keys (ignored): {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
        
        if not missing_keys and not unexpected_keys:
            print(f"  ✓✓ Perfect match! All parameters loaded successfully.")
        
    except Exception as exc:
        print(f"[WARN] Could not load baseline checkpoint into VID_Trans: {exc}. Proceeding with backbone pretrain only.")
    vid_model.eval()

    # Extract features with verbose mode to verify dense sampling
    print("\n" + "="*80)
    print("Extracting baseline features (query) with DENSE sampling...")
    if args.legacy_eval:
        print("[MODE] Legacy evaluation: using original VID_Test.py camid collection")
    else:
        print("[MODE] Correct evaluation: using one camid per tracklet")
    print("="*80)
    qf, q_pids, q_camids = extract_vidtrans_features(vid_model, q_set, device, verbose=True, legacy_camid=args.legacy_eval)
    print(f"✓ Query features extracted: {qf.shape}")
    
    print("\n" + "="*80)
    print("Extracting baseline features (gallery) with DENSE sampling...")
    print("="*80)
    gf, g_pids, g_camids = extract_vidtrans_features(vid_model, g_set, device, verbose=True, legacy_camid=args.legacy_eval)
    print(f"✓ Gallery features extracted: {gf.shape}")

    # Compute baseline distance matrix and metrics
    print("Computing baseline distances and metrics...")
    dist_base = compute_euclidean_dist(qf, gf)
    cmc_base, mAP_base = evaluate(dist_base, q_pids, g_pids, q_camids, g_camids)
    print("Baseline Results ----------")
    print(f"mAP: {mAP_base:.1%}")
    print(f"Rank-1: {cmc_base[0]:.1%}")

    # Stage-2: Hierarchical re-ranker
    print("\n" + "="*80)
    print("STAGE 2: Loading Hierarchical Cross-Attention Re-ranker")
    print("="*80)
    print(f"Loading re-ranker checkpoint from: {args.hierarchical_ckpt}")
    hier_obj = safe_torch_load(args.hierarchical_ckpt, map_location="cpu")
    
    # 显示checkpoint信息
    if isinstance(hier_obj, dict):
        print(f"\n[Checkpoint Info]")
        if "epoch" in hier_obj:
            print(f"  Epoch: {hier_obj['epoch']}")
        if "val_auc" in hier_obj or "best_auc" in hier_obj:
            auc = hier_obj.get('val_auc', hier_obj.get('best_auc', 0))
            print(f"  Validation AUC: {auc:.4f}")
        if "val_acc" in hier_obj:
            print(f"  Validation Acc: {hier_obj['val_acc']:.4f}")
    
    # Resolve to plain state_dict (support {'model': ...} or {'state_dict': ...})
    if isinstance(hier_obj, dict) and "model" in hier_obj:
        hier_state = hier_obj["model"]
    elif isinstance(hier_obj, dict) and "state_dict" in hier_obj:
        hier_state = hier_obj["state_dict"]
    else:
        hier_state = hier_obj

    # Infer camera_num from checkpoint if possible
    ckpt_cam_num = infer_camera_num_from_state_dict(hier_state)
    use_cam_num = ckpt_cam_num if ckpt_cam_num is not None else camera_num
    print(f"[INFO] Dataset camera_num={camera_num}; Checkpoint camera_num={ckpt_cam_num}; Using camera_num={use_cam_num}")

    print("Initializing Hierarchical Cross-Attention re-ranker...")
    # 注意：pretrained_path=None，我们将从 hierarchical checkpoint 加载全部权重
    rerank_model = HierarchicalCrossAttentionReID(
        img_size=(256, 128),
        embed_dim=768,
        num_heads=12,
        stride_size=16,
        camera_num=use_cam_num,
        pretrained_path=None,  # 不加载ImageNet预训练，改为从hierarchical checkpoint加载
    ).to(device)

    model_state = rerank_model.state_dict()

    if isinstance(hier_state, dict):
        hier_state = {(k[7:] if k.startswith('module.') else k): v for k, v in hier_state.items()}

    print(f"\n[Loading Parameters]")
    loaded_state = None
    try:
        missing, unexpected = rerank_model.load_state_dict(hier_state, strict=True)
        loaded_state = hier_state
    except RuntimeError as exc:
        print(f"  [WARN] Strict load failed: {exc}")
        compatible_state = {
            k: v
            for k, v in hier_state.items()
            if k in model_state and getattr(v, 'shape', None) == model_state[k].shape
        }
        missing, unexpected = rerank_model.load_state_dict(compatible_state, strict=False)
        loaded_state = compatible_state
        print(f"  [WARN] Loaded shape-compatible subset only.")

    print(f"  ✓ Loaded {len(loaded_state)} parameters from hierarchical checkpoint")
    if missing:
        print(f"  ⚠️  Missing {len(missing)} keys (will use random init): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  ⚠️  Unexpected {len(unexpected)} keys (ignored): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if not missing and not unexpected:
        print(f"  ✓✓ Perfect match! All parameters loaded successfully.")

    rerank_model.eval()
    print(f"  ✓ Re-ranker model ready!")

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
