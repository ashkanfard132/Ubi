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
from utils import make_balanced_test, get_loss, get_optimizer, get_scheduler, set_seed
from train_eval_test import train_and_evaluate, train_model, evaluate_model
from model import get_torch_model, get_ml_model, PretrainedClassifierHead
from visualization import (
  plot_all_results, visualize_distribution, plot_training_curves, plot_eval_overlays)



args = get_args()

print("\n========== ARGUMENTS ==========")
for k, v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("================================\n")

set_seed(args.seed)

# WandB Setup
wandb_run = None
_HAS_WB_SK = False
if args.wandb:
    import wandb
    # >>> optional sklearn plot helpers
    try:
        from wandb.sklearn import plot_confusion_matrix as wb_conf_mat
        from wandb.sklearn import plot_roc as wb_plot_roc
        from wandb.sklearn import plot_precision_recall as wb_plot_pr
        _HAS_WB_SK = True
    except Exception:
        _HAS_WB_SK = False

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
    feature_groups = feature_groups1 
else:
    Xf, Xs, y = Xf1, Xs1, y1
    feature_groups = feature_groups1

# Visualize label distribution
if args.plots and 'distribution' in args.plots:
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
    encode_sequence_windows,          
    make_balanced_test,
    oversample, undersample, smote_sample, smotee_sample,
    get_loss, get_optimizer, get_scheduler,
    train_model, evaluate_model,
    wandb_run=wandb_run
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
history = results.get("history", None)

# Plot training curves (train vs val vs val_bal) once.
if history and (args.plots is None or 'curves' in args.plots):
    plot_training_curves(history, out_dir="results", prefix=f"{args.model}_", wandb=wandb_run)


if args.plots:
    plot_all_results(
        Xf_test, y_test, y_pred_nat, y_prob_nat, feature_groups,
        prefix="", plot_dir="results", wandb=wandb_run, plots_to_make=args.plots, history=None
    )
    plot_all_results(
        Xf_test_bal, y_test_bal, y_pred_bal, y_prob_bal, feature_groups,
        prefix="bal_", plot_dir="results", wandb=wandb_run, plots_to_make=args.plots, history=None
    )

    if y_test is not None and y_prob_nat is not None and y_test_bal is not None and y_prob_bal is not None:
        plot_eval_overlays(
            y_nat=y_test, p_nat=y_prob_nat,
            y_bal=y_test_bal, p_bal=y_prob_bal,
            out_dir="results", prefix=f"{args.model}_", wandb=wandb_run
        )

if args.wandb and _HAS_WB_SK:

    y_prob_nat = np.asarray(y_prob_nat).reshape(-1)   
    y_prob_bal = np.asarray(y_prob_bal).reshape(-1)   
    assert y_test.shape[0] == y_prob_nat.shape[0], "y_test and y_prob_nat length mismatch"
    assert y_test_bal.shape[0] == y_prob_bal.shape[0], "y_test_bal and y_prob_bal length mismatch"
    prob2_nat = np.column_stack([1.0 - y_prob_nat, y_prob_nat])  
    prob2_bal = np.column_stack([1.0 - y_prob_bal, y_prob_bal]) 

    # Confusion matrices
    try:
        wb_conf_mat(y_test, y_pred_nat, labels=["neg", "pos"])
        wb_conf_mat(y_test_bal, y_pred_bal, labels=["neg", "pos"])
    except Exception as e:
        wandb.log({"warn/conf_mat_error": str(e)})

    # ROC + PR with explicit labels that match prob column order
    try:
        wb_plot_roc(y_test, prob2_nat, labels=["neg", "pos"])
        wb_plot_roc(y_test_bal, prob2_bal, labels=["neg", "pos"])
        wb_plot_pr(y_test, prob2_nat, labels=["neg", "pos"])
        wb_plot_pr(y_test_bal, prob2_bal, labels=["neg", "pos"])
    except Exception as e:
        wandb.log({"warn/roc_pr_error": str(e)})

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
excel_path = f"results/{args.model}_{args.features}_{args.window_size}_{timestamp}_metrics.xlsx"

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

if args.wandb:
    wandb_run.log({f"nat_{k}": v for k, v in metrics_nat.items()})
    wandb_run.log({f"bal_{k}": v for k, v in metrics_bal.items()})
    # summaries (show up on run page right sidebar)
    if "f1" in metrics_nat:
        wandb.run.summary["final/natural_f1"] = float(metrics_nat["f1"])
    if "f1" in metrics_bal:
        wandb.run.summary["final/balanced_f1"] = float(metrics_bal["f1"])
