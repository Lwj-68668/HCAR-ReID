# Robust Video Person Re-Identification via Hierarchical Cross-Attention Mechanisms

Official implementation for the manuscript currently submitted to *The Visual Computer*.

## Mandatory Manuscript Linkage Statement

This repository and its archived release are directly related to the manuscript currently submitted to *The Visual Computer*:

`Robust Video Person Re-Identification via Hierarchical Cross-Attention Mechanisms`

If you use this code, derived implementations, or released artifacts, please cite the manuscript.

## Persistent Resource Links (Fill Before Final Submission)

- GitHub repository: `<GITHUB_REPOSITORY_URL>`
- Code archive DOI (Zenodo): `<CODE_DOI_URL>`
- Data DOI or data statement link: `<DATA_DOI_OR_DATA_STATEMENT_URL>`

Note: This project uses standard public ReID benchmarks. If dataset redistribution is restricted, publish dataset access instructions and your processed split metadata with a DOI instead of rehosting raw images.

## Repository Structure

- `train_hierarchical.py`: Training entry for hierarchical cross-attention pair matching.
- `train_baseline.py`: Training entry for baseline cross-attention pair matching.
- `CrossAttention_model.py`: Baseline bidirectional cross-attention matcher.
- `HierarchicalCrossAttention_model.py`: Hierarchical (spatial + temporal) cross-attention matcher.
- `pair_losses.py`: Focal BCE and symmetric KL regularizer used by hierarchical training.
- `Dataloader.py`: Dataset wrappers, pair construction, deterministic loader setup.
- `rerank_evaluate.py`: Two-stage re-ranking evaluation.
- `VID_Trans_ReID.py`, `VID_Test.py`: Legacy full retrieval pipeline.
- `Datasets/`: Dataset readers for MARS, iLIDS-VID, PRID2011.
- `loss/`, `Loss_fun.py`: Legacy ID/triplet/center-loss modules.
- `docs/`: Reproducibility, dataset, DOI, and manuscript-ready text templates.

## Environment and Dependencies

Recommended:

- Python `3.10` or `3.11`
- CUDA-capable GPU for training

Install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install `torch`/`torchvision` first from the official PyTorch index, then run `pip install -r requirements.txt`.

## Data Preparation

Supported datasets:

- `Mars`
- `iLIDSVID`
- `PRID`

Detailed folder requirements are in `docs/DATASET_PREPARATION.md`.

## Pretrained Backbone

Pair-matching scripts expect a ViT checkpoint path (default):

- `jx_vit_base_p16_224-80ecf9dd.pth`

Place it at repository root or pass `--pretrained_path` explicitly.

## Training and Evaluation

1) Hierarchical pair matcher (recommended):

```bash
python train_hierarchical.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_hierarchical --seed 42
```

2) Optional robust loss settings:

```bash
python train_hierarchical.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_hierarchical --seed 42 --use_focal --focal_alpha 0.5 --focal_gamma 2.0 --symkl_w 0.1
```

3) Baseline pair matcher:

```bash
python train_baseline.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_baseline --seed 42
```

4) Two-stage re-ranking evaluation:

```bash
python rerank_evaluate.py --Dataset_name Mars --baseline_ckpt checkpoints_baseline\\baseline_best_auc_XXXX.pth --hierarchical_ckpt checkpoints_hierarchical\\hierarchical_best_auc_XXXX.pth --seq_len 4 --topk 50 --device cuda
```

5) Legacy full pipeline:

```bash
python VID_Trans_ReID.py --Dataset_name Mars
python VID_Test.py --Dataset_name Mars --model_path Mars_best.pth
```

## Reproducibility and Algorithm Documentation

- Reproducibility protocol: `docs/REPRODUCIBILITY.md`
- Dataset preparation details: `docs/DATASET_PREPARATION.md`
- Manuscript-to-code mapping: `docs/ALGORITHM_MAPPING.md`
- DOI and release workflow: `docs/DOI_AND_RELEASE.md`
- Manuscript-ready text snippets: `docs/MANUSCRIPT_SNIPPETS.md`
- Editor response template: `docs/EDITOR_RESPONSE_TEMPLATE.md`

## Citation

Machine-readable citation metadata is provided in `CITATION.cff`.

Please replace placeholders in `CITATION.cff` and this README with your final title/authors/DOI before public release.
