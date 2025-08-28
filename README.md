# Ubi — Ubiquitination Site Prediction

End-to-end pipeline for predicting ubiquitination sites from protein sequences using both **classical ML** (Random Forest, XGBoost, etc.) and **deep learning** (CNN/LSTM/Transformer, ProtBert, ESM2). Train on one plant or merge two plants, optionally add **PSSM** features, handle class imbalance, run cross-validation, and save rich plots and metrics.

---

## Features

- **Two pipelines**
  - **ML / Tabular features:** `rf`, `xgb`, `ada`, `cat`, `svm`, `lreg`, `mlp`
  - **DL / Sequence models:** `cnn`, `lstm`, `gru`, `transformer`, `prot_bert`, `distil_prot_bert`, `esm2_t6_8m`
- **Inputs**
  - Plant 1 (**required**): `--fasta` + `--excel`
  - Plant 2 (**optional merge**): `--fasta2` + `--excel2`
  - Optional **PSSM** folders aligned to FASTA order: `--pssm1`, `--pssm2`
- **Feature families (ML / MLP)**
  - `aac`, `dpc`, `tpc`, `pssm`, `physicochem`, `binary`
- **Imbalance handling:** `over`, `under`, `smote`, `smotee`, `none`
- **Cross-validation:** `--cv k` (k-fold) or `--cv 1` (single split)
- **Visualization:** distribution, confusion, ROC, PR, PCA/feature plots, training curves
- **Outputs:** JSON & Excel metrics + PNGs in `results/`
- **Optional W&B logging:** `--wandb` (plus project/entity)

---

## Installation

```bash
git clone https://github.com/ashkanfard132/Ubi.git
cd Ubi

# Core scientific stack
pip install numpy pandas scikit-learn imbalanced-learn tqdm matplotlib seaborn openpyxl

# ML extras
pip install xgboost catboost

# Deep learning
pip install torch torchvision torchaudio

# Protein language models (if you plan to use them)
pip install transformers     # for ProtBert/DistilProtBert
pip install fair-esm         # for ESM2 models

# Optional experiment tracking
pip install wandb
