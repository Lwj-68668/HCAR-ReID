# HCAR-ReID v1.0.0

Initial public release of the codebase associated with the manuscript currently submitted to *The Visual Computer*:

**Robust Video Person Re-Identification via Hierarchical Cross-Attention Mechanisms**

## Manuscript Linkage

This release is directly related to the manuscript above.  
If you use this code or derived resources, please cite the manuscript.

## What's Included

- Hierarchical cross-attention pair-matching training pipeline (`train_hierarchical.py`)
- Baseline cross-attention pair-matching training pipeline (`train_baseline.py`)
- Two-stage re-ranking evaluation (`rerank_evaluate.py`)
- Legacy full retrieval pipeline (`VID_Trans_ReID.py`, `VID_Test.py`)
- Dataset loaders for MARS, iLIDS-VID, and PRID2011 (`Datasets/`)
- Reproducibility and release documentation (`docs/`)
- Citation metadata (`CITATION.cff`)

## Reproducibility Notes

- Dependency list: `requirements.txt`
- Environment and usage: `README.md`
- Dataset layout instructions: `docs/DATASET_PREPARATION.md`
- Reproduction protocol: `docs/REPRODUCIBILITY.md`

## Quick Start

```bash
python -m venv .venv
pip install -r requirements.txt
python train_hierarchical.py --Dataset_name Mars --epochs 50 --batch_size 32 --seq_len 4 --save_dir checkpoints_hierarchical --seed 42
```

## DOI and Citation

- GitHub repository: `<GITHUB_REPOSITORY_URL>`
- Code DOI (Zenodo): `<CODE_DOI_URL>`
- Data DOI / data statement: `<DATA_DOI_OR_DATA_STATEMENT_URL>`

Please replace the placeholders above once DOI minting is completed.
