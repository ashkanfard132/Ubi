````markdown
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
````

> GPU (CUDA) is recommended for deep learning models. Classical sklearn models run on CPU.

---

## Data

* **FASTA**: protein sequences for Plant 1 (and optional Plant 2).
* **Excel**: labels/metadata expected by the repository’s loader.
* **PSSM** (optional): folders with PSSM features **in the same order** as sequences in the corresponding FASTA.

---

## Quick Start

### ML pipeline (two plants + PSSM + mixed features)

```bash
python -u main.py \
  --model rf \
  --fasta "/path/Plant1.fasta" \
  --excel "/path/Plant1.xlsx" \
  --fasta2 "/path/Plant2.fasta" \
  --excel2 "/path/Plant2.xlsx" \
  --pssm1 "/path/PSSM_Plant1" \
  --pssm2 "/path/PSSM_Plant2" \
  --features aac dpc tpc pssm physicochem binary \
  --window_size 21 \
  --sampling none --sampling_ratios 1.0 \
  --cv 1 \
  --plots distribution confusion roc pr
```

### Deep Learning (sequence CNN)

```bash
python -u main.py \
  --model cnn \
  --fasta "/path/Plant1.fasta" \
  --excel "/path/Plant1.xlsx" \
  --window_size 21 \
  --device cuda \
  --epochs 20 --batch_size 64 --batch_size_val 64 \
  --loss bce --optim adam --lr 1e-3 --sched step --step_size 10 --gamma 0.1 \
  --sampling over --sampling_ratios 1.0 \
  --cv 1 \
  --plots distribution curves confusion roc pr
```

---

## Command-line Arguments (high-level)

**Required**

* `--model`
  ML: `rf`, `xgb`, `ada`, `cat`, `svm`, `lreg`, `mlp`
  DL: `cnn`, `lstm`, `gru`, `transformer`, `prot_bert`, `distil_prot_bert`, `esm2_t6_8m`
* `--fasta`, `--excel` (Plant 1)

**Optional data**

* `--fasta2`, `--excel2` (Plant 2, merged with Plant 1)
* `--pssm1`, `--pssm2` (PSSM directories aligned to FASTA order)

**Features (ML/MLP)**

* `--features aac dpc tpc [pssm physicochem binary]` (default: `aac dpc tpc`)

**Window / context**

* `--window_size` (odd integer 5–35; default 21)

**Imbalance**

* `--sampling {over,under,smote,smotee,none}`
* `--sampling_ratios ...` (must match length of `--sampling`)

**Cross-validation**

* `--cv 1` (single split)
* `--cv k` (k-fold, metrics averaged)

**Threshold tuning**

* `--best_threshold` (learn F2-optimal threshold on validation; primarily for ML, supported in code for DL too)

**Training (DL & MLP only)**

* `--epochs`, `--batch_size`, `--batch_size_val`
* `--device {cpu|cuda}`
* `--loss {bce,mse,mae,focal}` + `--pos_weight`
* `--optim {adam,sgd,rmsprop,adamw,amsgrad}`, `--lr`, `--weight_decay`
* `--sched {step,exp,plateau,cosine,none}`, `--step_size`, `--gamma`, `--t_max`
* `--dropout`
* `--freeze_pretrained` (ProtBert/ESM2 feature-extraction mode)

**Visualization**

* `--plots` any of:

  ```
  distribution curves confusion roc pr
  feature_distribution feature_correlation feature_scatter
  boxplot violinplot groupwise_pca overlays
  ```

**W\&B logging (optional)**

* `--wandb` to enable
* `--wandb_project` (default `ubiquitination-prediction`)
* `--wandb_entity` (defaults to `WANDB_ENTITY` env; only required if `--wandb` is set)
* `--wandb_api_key` (optional, for notebook login)
* To force-disable: `export WANDB_DISABLED=true`

**Reproducibility**

* `--seed 42` (default)

**SVM grid (optional)**

* `--svm-grid`
* `--svm-kernels rbf,linear,poly`
* `--svm-C 0.1,1,5,10`
* `--svm-gamma scale,auto,0.001,0.0001`

> **Note:** For sklearn models (`rf/xgb/ada/cat/svm/lreg`) the flags for epochs/optimizer/device are ignored. They apply to **MLP/DL** only.

---

## Outputs (in `results/`)

* `results_{model}_{YYYYMMDD_HHMMSS}.json` — metrics (natural test)
* `results_{model}_{timestamp}_bal.json` — metrics (balanced test)
* `{model}_{features_or_seq}_{window}_{timestamp}_metrics.xlsx` — Excel summary
* Plot PNGs: `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, optional `*_curve.png` for training, plus feature/PCA overlays

---

## Tips & Troubleshooting

* Keep `--window_size` **odd** (e.g., 5, 9, 21, 31).
* Ensure PSSM directories match FASTA **order** exactly.
* For transformer/ProtBert/ESM2, watch GPU memory; reduce `--batch_size` or `--window_size` if you hit OOM.
* For heavy class imbalance, try `--sampling over` or `--sampling smote` (feature-based models).
* Excel export requires `openpyxl`.

---

## Project Structure (key files)

```
Ubi/
├─ main.py                 # Entry: parse args, train/eval, save results
├─ args.py                 # All CLI arguments
├─ data_preprocessing.py   # Loading, feature extraction, PSSM parsing, sampling
├─ train_eval_test.py      # Training loops, CV, evaluation, threshold search
├─ model.py                # Model factories (ML & DL, ProtBert/ESM2 heads)
├─ utils.py                # Metrics, optimizers, schedulers, helpers
├─ visualization.py        # Plots: confusion/ROC/PR/PCA/curves/overlays
└─ results/                # Created at runtime (metrics & figures)
```

---

## Citation

If you use this code in a publication, please cite this repository and any pre-trained models you employ (ProtBert/ESM2). Add your BibTeX here.

---
