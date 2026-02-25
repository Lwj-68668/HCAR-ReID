# Algorithm-to-Code Mapping

This document maps manuscript components to concrete implementation files in this repository.

## Core Pair-Matching Models

| Manuscript Component | Code Location | Implementation Notes |
| --- | --- | --- |
| Baseline bidirectional cross-attention matcher | `CrossAttention_model.py` (`CrossAttentionReID`) | Shared TransReID backbone, flatten patch tokens across time, bidirectional cross-attention, MLP binary head. |
| Hierarchical cross-attention matcher | `HierarchicalCrossAttention_model.py` (`HierarchicalCrossAttentionReID`) | Two stages: spatial cross-attention per frame, then temporal cross-attention over frame-level features. |
| Pairwise training objective | `train_baseline.py`, `train_hierarchical.py` | BCEWithLogits on positive/negative video pairs. |
| Robust loss variants | `pair_losses.py` | Focal BCE (`FocalBCEWithLogitsLoss`) and symmetric KL consistency regularization. |

## Data and Pair Construction

| Manuscript Component | Code Location | Implementation Notes |
| --- | --- | --- |
| Dataset parsing (MARS/iLIDS-VID/PRID2011) | `Datasets/` | Builds train/query/gallery tracklets with dataset-specific directory and split logic. |
| Pair sampling for matching | `Dataloader.py` (`VideoPairDataset`) | Generates balanced positive/negative video pairs for training. |
| Validation pair protocol | `Dataloader.py` (`QueryGalleryPairDataset`) | Deterministic query-gallery pair construction with center sampling. |
| Deterministic data loading | `Dataloader.py` (`_seed_worker`, `make_pair_dataloader`) | Worker seeds and data loader generator seeded for reproducibility. |

## Two-Stage Retrieval and Re-Ranking

| Manuscript Component | Code Location | Implementation Notes |
| --- | --- | --- |
| Stage-1 global retrieval | `VID_Trans_model.py`, `VID_Test.py` | Extracts video-level descriptors and computes initial ranking with Euclidean distance. |
| Stage-2 pairwise re-scoring | `rerank_evaluate.py` | Uses hierarchical pair model to rescore top-k candidate pairs and fuse scores. |

## Hierarchical Forward Pass (Conceptual)

1. Extract per-frame patch tokens from both videos with a shared TransReID backbone.
2. Remove `[CLS]` token and keep patch tokens.
3. Apply bidirectional spatial cross-attention frame-by-frame.
4. Average pool patches to obtain frame-level features.
5. Apply bidirectional temporal cross-attention over frame features.
6. Average pool time dimension to obtain video-level features.
7. Concatenate two video features and predict match logit with a binary classifier head.

This process is implemented in `HierarchicalCrossAttention_model.py` (`forward`).
