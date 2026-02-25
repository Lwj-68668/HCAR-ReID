# Reproducibility Protocol

This protocol is intended for paper revision, release preparation, and third-party verification.

## 1) Record Environment

Before running experiments, record:

- OS and version
- Python version
- GPU model and CUDA version
- `pip freeze` output
- Git commit hash

Recommended commands:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
git rev-parse HEAD
pip freeze > freeze.txt
```

## 2) Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Prepare Data

Follow `docs/DATASET_PREPARATION.md`.

## 4) Baseline Pair-Matching Run

```bash
python train_baseline.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_baseline --seed 42
```

Expected artifact:

- `checkpoints_baseline/baseline_best_auc_*.pth`

## 5) Hierarchical Pair-Matching Run

```bash
python train_hierarchical.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_hierarchical --seed 42
```

Optional robust setting:

```bash
python train_hierarchical.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_hierarchical --seed 42 --use_focal --focal_alpha 0.5 --focal_gamma 2.0 --symkl_w 0.1
```

Expected artifact:

- `checkpoints_hierarchical/hierarchical_best_auc_*.pth`

## 6) Two-Stage Re-Ranking Evaluation

```bash
python rerank_evaluate.py --Dataset_name Mars --baseline_ckpt checkpoints_baseline\\baseline_best_auc_XXXX.pth --hierarchical_ckpt checkpoints_hierarchical\\hierarchical_best_auc_XXXX.pth --seq_len 4 --topk 50 --device cuda
```

## 7) Legacy Pipeline (Optional)

```bash
python VID_Trans_ReID.py --Dataset_name Mars
python VID_Test.py --Dataset_name Mars --model_path Mars_best.pth
```

## 8) What to Report in the Paper/Release

- Exact training command lines
- Dataset split used
- Random seed value
- Best checkpoint filename
- Rank-1 and mAP results
- Hardware and training time

## 9) Recommended Release Artifacts

- Source code snapshot (tagged release)
- Final config/command logs
- Trained checkpoint hashes (SHA256)
- Evaluation output logs
- `CITATION.cff`
