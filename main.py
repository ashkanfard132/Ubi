import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

from args import get_args
from data_preprocessing import (
    load_dataset, AMINO_ACIDS, parse_pssm_folder_by_order, encode_sequence_windows,
    oversample, undersample, smote_sample, smotee_sample
)
from utils import make_balanced_test, get_loss, get_optimizer, get_scheduler
from train_eval_test import train_and_evaluate, train_model, evaluate_model
from model import get_torch_model, get_ml_model, PretrainedClassifierHead
from visualization import plot_all_results, visualize_distribution
# from visualization import (
#     visualize_distribution,
#     visualize_confusion_matrix,
#     visualize_roc,
#     visualize_precision_recall,
#     visualize_groupwise_pca,
#     visualize_feature_distribution,
#     visualize_feature_correlation,
#     visualize_feature_scatter,
#     visualize_boxplot,
#     visualize_violinplot
# )

args = get_args()

print("\n========== ARGUMENTS ==========")
for k, v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("================================\n")

# Only import wandb if needed
wandb_run = None
if args.wandb:
    import wandb
    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        group=args.model,
        name=run_name
    )
    wandb_run = wandb

# Load PSSM data
pssm1 = parse_pssm_folder_by_order(args.fasta, args.pssm1) if args.pssm1 else None
pssm2 = parse_pssm_folder_by_order(args.fasta2, args.pssm2) if args.fasta2 and args.pssm2 else None


# Load datasets
print("Loading dataset for Plant 1...", flush=True)
Xf1, Xs1, y1, feature_groups1 = load_dataset(
    args.fasta, args.excel,
    selected_features=args.features,
    pssm_data=pssm1,
    window_size=args.window_size
)

if args.fasta2 and args.excel2:
    print("Loading dataset for Plant 2...", flush=True)
    Xf2, Xs2, y2, feature_groups2 = load_dataset(
        args.fasta2, args.excel2,
        selected_features=args.features,
        pssm_data=pssm2,
        window_size=args.window_size
    )
    Xf = np.vstack([Xf1, Xf2])
    Xs = np.vstack([Xs1, Xs2])
    y = np.concatenate([y1, y2])
    feature_groups = feature_groups1  # Assume features are the same; use Plant 1's mapping
else:
    Xf, Xs, y = Xf1, Xs1, y1
    feature_groups = feature_groups1

# print("Xf shape:", Xf.shape)
# print("Sum of 'pssm' feature columns in Xf:", np.sum(Xf[:, feature_groups['PSSM']]))
# print("Min/max of PSSM features:", np.min(Xf[:, feature_groups['PSSM']]), np.max(Xf[:, feature_groups['PSSM']]))

# print("Positives in y:", np.sum(y))
# print("Negatives in y:", np.sum(y == 0))

# print("Any NaN in Xf:", np.isnan(Xf).any())
# print("Any Inf in Xf:", np.isinf(Xf).any())


# Visualize label distribution
if args.plots:
    os.makedirs("results", exist_ok=True)
    visualize_distribution(y, "results/label_distribution.png")
    if args.wandb:
        wandb.log({"label_distribution": wandb.Image("results/label_distribution.png")})


# Visualizations for NATURAL (original) test set
# --------- ALL CV/SPLIT/TRAIN/EVAL LOGIC IN ONE CALL -----------
results = train_and_evaluate(
    args,
    Xf, Xs, y, feature_groups,
    get_torch_model, get_ml_model, PretrainedClassifierHead,
    encode_sequence_windows,          # <--- add these lines
    make_balanced_test,
    oversample, undersample, smote_sample, smotee_sample,
    get_loss, get_optimizer, get_scheduler,
    train_model, evaluate_model
)

metrics_nat = results["metrics_nat"]
metrics_bal = results["metrics_bal"]
y_test = results["y_test"]
y_pred_nat = results["y_pred_nat"]
y_prob_nat = results["y_prob_nat"]
y_test_bal = results["y_test_bal"]
y_pred_bal = results["y_pred_bal"]
y_prob_bal = results["y_prob_bal"]
Xf_test = results.get("Xf_test", None)
Xf_test_bal = results.get("Xf_test_bal", None)

# ---------------------------------------------------------------

# Visualizations for NATURAL (original) test set

if args.plots:
    if y_test is not None and y_pred_nat is not None and y_prob_nat is not None:
        plot_all_results(
            Xf_test,
            y_test,
            y_pred_nat,
            y_prob_nat,
            feature_groups,
            prefix="",
            plot_dir="results",
            wandb_run=wandb_run
        )
    if y_test_bal is not None and y_pred_bal is not None and y_prob_bal is not None:
        plot_all_results(
            Xf_test_bal,
            y_test_bal,
            y_pred_bal,
            y_prob_bal,
            feature_groups,
            prefix="bal_",
            plot_dir="results",
            wandb_run=wandb_run
        )



# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("results", exist_ok=True)
results_path_nat = f"results/results_{args.model}_{timestamp}.json"
results_path_bal = f"results/results_{args.model}_{timestamp}_bal.json"
with open(results_path_nat, "w") as f:
    json.dump(metrics_nat, f, indent=4)
with open(results_path_bal, "w") as f:
    json.dump(metrics_bal, f, indent=4)


# Output
print("\nEvaluation Results (NATURAL TEST SET):")
for k, v in metrics_nat.items():
    print(f"{k.capitalize()}: {v:.4f}")
print(f"\nSaved metrics to {results_path_nat}")

print("\nEvaluation Results (BALANCED TEST SET):")
for k, v in metrics_bal.items():
    print(f"{k.capitalize()}: {v:.4f}")
print(f"\nSaved balanced metrics to {results_path_bal}")

# --- Save metrics to Excel ---
# derive a simple “dataset” name from your input files
data_name = os.path.splitext(
    os.path.basename(args.excel if args.excel else args.fasta)
)[0]

# make a metrics DataFrame
metrics_df = pd.DataFrame({
    'Natural': metrics_nat,
    'Balanced': metrics_bal
})
metrics_df.index.name = 'Metric'

# Excel file path includes data name, model, and timestamp
excel_path = f"results/{data_name}_{args.model}_{timestamp}_metrics.xlsx"

with pd.ExcelWriter(excel_path) as writer:
    # sheet 1: the metrics table
    metrics_df.to_excel(writer, sheet_name='Metrics')
    # sheet 2: summary info
    info_df = pd.DataFrame({
        'Model':     [args.model],
        'Dataset':   [data_name],
        'Timestamp': [timestamp]
    })
    info_df.to_excel(writer, sheet_name='Info', index=False)

print(f"\nSaved all metrics to Excel: {excel_path}")

if wandb_run:
    wandb_run.log({f"nat_{k}": v for k, v in metrics_nat.items()})
    wandb_run.log({f"bal_{k}": v for k, v in metrics_bal.items()})