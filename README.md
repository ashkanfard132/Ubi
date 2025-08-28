````markdown
# Ubi — Ubiquitination Site Prediction

End-to-end pipeline for predicting ubiquitination sites from protein sequences using both **classical ML** (Random Forest, XGBoost, etc.) and **deep learning** (CNN/LSTM/Transformer, ProtBert, ESM2). Supports single-plant or two-plant training, optional **PSSM** features, multiple feature families, class-imbalance handling, cross-validation, rich plots, and optional Weights & Biases logging.

---

## ✨ Features

- **Two pipelines**
  - **ML (tabular features):** `rf`, `xgb`, `ada`, `cat`, `svm`, `lreg`, `mlp`
  - **DL (sequence models):** `cnn`, `lstm`, `gru`, `transformer`, `prot_bert`, `distil_prot_bert`, `esm2_t6_8m`
- **Data inputs**
  - Plant 1 (**required**): `--fasta` + `--excel`
  - Plant 2 (**optional, merged**): `--fasta2` + `--excel2`
  - Optional **PSSM** folders aligned to FASTA order: `--pssm1`, `--pssm2`
- **Feature families (ML / MLP)**
  - `aac`, `dpc`, `tpc`, `pssm`, `physicochem`, `binary`
- **Imbalance handling:** `over`, `under`, `smote`, `smotee`, `none`
- **Cross-validation:** `--cv k` (k-fold) or `--cv 1` (single split)
- **Visualization:** distribution, confusion, ROC, PR, PCA-based feature plots, training curves
- **Outputs:** JSON + Excel metrics, plot PNGs under `results/`
- **Optional W&B logging:** `--wandb` (+ project/entity)

---

## 🧩 Installation

```bash
git clone https://github.com/ashkanfard132/Ubi.git
cd Ubi
pip install -r requirements.txt
# If you plan to use ProtBert/DistilProtBert:
pip install transformers
# If you plan to use ESM2:
pip install fair-esm
# For Excel output:
pip install openpyxl
````

> GPU (CUDA) is recommended for deep learning models. Classical sklearn models run on CPU.

---

## 📦 Data

* **FASTA**: protein sequences for Plant 1 (and optional Plant 2)
* **Excel**: labels/metadata expected by the repository’s loader
* **PSSM** (optional): folders with PSSM features **in the same order** as sequences in the corresponding FASTA

---

## 🚀 Quick Start

### ML pipeline (example: two plants + PSSM + mixed features)

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

**Option for ML pipeline**

* **Model Config**: `--model {rf, xgb, ada, cat, svm, lreg}`

  * `rf` (Random Forest), `xgb` (XGBoost), `ada` (AdaBoost), `cat` (CatBoost), `svm` (Support Vector Machine), `lreg` (Logistic Regression)
* **Dataset**: `--fasta`, `--excel` (Plant 1); `--fasta2`, `--excel2` (optional Plant 2)
* **Features**: any combination of `aac dpc tpc pssm physicochem binary`
* **Window**: `--window_size` (**odd**, e.g., 5–35). Affects feature windows.
* **Imbalance**: `--sampling {over, under, smote, smotee, none}` + `--sampling_ratios`
* **Cross-validation**: `--cv 1` (single split) or `--cv k`
* **Threshold tuning** (ML with `predict_proba`): add `--best_threshold`
* **SVM grid** (optional): `--svm-grid --svm-kernels rbf,linear --svm-C 0.1,1,5,10 --svm-gamma scale,auto,0.001,0.0001`

> Note: sklearn models ignore `--device`, `--epochs`, `--lr`, etc.

---

### Deep Learning pipeline (sequence models)

#### Example A — CNN on windowed sequences

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

#### Example B — ProtBert (freeze encoder for speed)

```bash
python -u main.py \
  --model prot_bert \
  --fasta "/path/Plant1.fasta" \
  --excel "/path/Plant1.xlsx" \
  --window_size 31 \
  --device cuda \
  --epochs 5 --batch_size 8 --batch_size_val 8 \
  --optim adamw --lr 2e-5 --sched cosine --t_max 5 \
  --freeze_pretrained \
  --best_threshold \
  --plots curves roc pr
```

**DL options summary**

* **Models**: `mlp`, `cnn`, `lstm`, `gru`, `transformer`, `prot_bert`, `distil_prot_bert`, `esm2_t6_8m`
* **Inputs**

  * **mlp**: uses feature vectors (`--features ...`), can include `pssm`
  * **cnn/lstm/gru/transformer/prot\_bert/distil\_prot\_bert/esm2**: learn **directly from sequence**; handcrafted features/PSSM are ignored
* **Window/length**: `--window_size` (used for sequence windows and tokenizer `max_length`)
* **Training**: `--epochs`, `--batch_size`, `--device {cpu|cuda}`, `--loss`, `--optim`, `--lr`, `--weight_decay`, `--sched`, `--dropout`
* **Pretrained**: `--freeze_pretrained` for ProtBert/ESM2 feature extraction
* **Imbalance**: prefer `over/under` for sequence models; `smote/smotee` are feature-space techniques
* **Threshold**: `--best_threshold` learns F2-optimal threshold on validation and applies to test

---

## 📊 Outputs

Saved under `results/`:

* `results_{model}_{YYYYMMDD_HHMMSS}.json` — metrics on natural test split
* `results_{model}_{timestamp}_bal.json` — metrics on balanced test split
* `{model}_{features_or_seq}_{window}_{timestamp}_metrics.xlsx` — Excel summary
* Plot PNGs: `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, optional training `*_curve.png`, PCA/feature plots, overlays

---

## 📈 Visualization flags

Use `--plots` with any of:

```
distribution curves confusion roc pr
feature_distribution feature_correlation feature_scatter
boxplot violinplot groupwise_pca overlays
```

* `curves` (DL/MLP): logs train/val/val\_bal metrics across epochs
* `overlays`: PR/ROC overlays comparing natural vs balanced test sets

---

## 🧪 Weights & Biases (optional)

* Enable: add `--wandb` (plus `--wandb_project`, `--wandb_entity` or env `WANDB_ENTITY`)
* Disable: simply **omit** `--wandb`. You can also set `WANDB_DISABLED=true`.

---

## ⚙️ Important arguments (quick reference)

* **Data**: `--fasta`, `--excel` \[required]; `--fasta2`, `--excel2` \[optional merge]; `--pssm1`, `--pssm2` \[optional]
* **Features**: `--features aac dpc tpc [pssm physicochem binary]` (default: `aac dpc tpc`)
* **Window**: `--window_size` (odd, 5–35; default 21)
* **Model**: `--model` (see lists above)
* **Training (DL/MLP)**: `--epochs`, `--batch_size`, `--batch_size_val`, `--device`, `--loss`, `--pos_weight`, `--optim`, `--lr`, `--weight_decay`, `--sched`, `--step_size`, `--gamma`, `--t_max`, `--dropout`, `--freeze_pretrained`
* **Imbalance**: `--sampling ...` + `--sampling_ratios ...`
* **Evaluation**: `--cv`, `--best_threshold`, `--plots ...`
* **Reproducibility**: `--seed` (default 42)

---

## 🧠 Tips & troubleshooting

* Keep `--window_size` **odd** (e.g., 5, 9, 21, 31).
* Ensure PSSM folders align with FASTA **order**; mismatches will raise issues during parsing.
* For transformer/BERT/ESM models, watch GPU memory; reduce `--batch_size` or `--window_size` on OOM.
* For heavy class imbalance, try `--sampling over` or `--sampling smote` (for feature-based models).
* Excel export requires `openpyxl`.

---

## 📂 Project structure (key files)

```
Ubi/
├─ main.py                 # Entry point: parsing + training/evaluation + saving results
├─ args.py                 # All CLI arguments and help
├─ data_preprocessing.py   # Loading datasets, feature extraction, PSSM parsing, sampling
├─ train_eval_test.py      # Train loops, k-fold CV, evaluation, threshold search
├─ model.py                # Model factories (ML & DL, incl. ProtBert/ESM2 heads)
├─ utils.py                # Metrics, optimizers, schedulers, helpers
├─ visualization.py        # Plots: confusion/ROC/PR/PCA/curves/overlays
├─ Ubiquitination_NoteBook.ipynb  # Example notebook (optional)
└─ results/                # Created at runtime with metrics and figures
```

---

## 🙌 Acknowledgments

* Protein language models: Hugging Face **ProtBert/DistilProtBert**, FAIR **ESM2**.
* Community packages: scikit-learn, XGBoost, CatBoost, imbalanced-learn, PyTorch, matplotlib/seaborn.

---

## 📣 Citation

If you use this code in a publication, please cite this repository and any pre-trained models you employ (ProtBert/ESM2). Add your BibTeX here.

---

## 🔗 License

Add a license file (e.g., `LICENSE`) and reference it here.

```

*Notes: repo file names and layout verified from the public listing; the arguments and behavior reflect the provided CLI and training code.* :contentReference[oaicite:0]{index=0}
::contentReference[oaicite:1]{index=1}
```
