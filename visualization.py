import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA

def safe_pca(X, n_components):
    n_samples, n_features = X.shape
    actual_n = min(n_components, n_samples, n_features)
    if actual_n < 2:
        raise ValueError(f"Cannot run PCA: n_samples={n_samples}, n_features={n_features}, need at least 2 for visualization.")
    return PCA(n_components=actual_n).fit_transform(X), actual_n

def _safe_save(fig, path):
    try:
        fig.savefig(path)
        assert os.path.isfile(path), f"File not found after save: {path}"
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        plt.close(fig)

def visualize_groupwise_pca(X, y, feature_groups, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    for group_name, indices in feature_groups.items():
        if len(indices) < 2:
            continue
        Xg = X[:, indices]
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(Xg)
        df = pd.DataFrame({
            'PC1': Xp[:,0], 'PC2': Xp[:,1], 'Label': y
        })

        fig = plt.figure(figsize=(6,5))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Label', alpha=0.6)
        plt.title(f'PCA Scatter: {group_name}')
        plt.tight_layout()
        path = os.path.join(out_dir, f"pca_scatter_{group_name.lower()}.png")
        _safe_save(fig, path)

        fig = plt.figure(figsize=(5,4))
        sns.violinplot(x='Label', y='PC1', data=df, inner='quartile')
        plt.title(f'PCA Violin (PC1): {group_name}')
        plt.tight_layout()
        path = os.path.join(out_dir, f"pca_violin_{group_name.lower()}.png")
        _safe_save(fig, path)

def visualize_distribution(y, save_path=None):
    fig = plt.figure(figsize=(5,3))
    sns.countplot(x=y)
    plt.title("Label Distribution")
    plt.xlabel("Class"); plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_roc(y_true, y_scores, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_precision_recall(y_true, y_scores, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    fig = plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall"); plt.legend(loc="lower left")
    plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_feature_scatter(X, y, save_path=None):
    Xr, n_components = safe_pca(X, 2)
    df = pd.DataFrame(Xr, columns=['PC1','PC2'])
    df['Label'] = y
    fig = plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Label', alpha=0.6)
    plt.title("PCA Scatter of Features"); plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_feature_distribution(X, y, save_path=None):
    Xr, n_components = safe_pca(X, 5)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(Xr, columns=cols)
    df['Label'] = y
    melted = df.melt(id_vars='Label', var_name='PC', value_name='Val')
    fig = plt.figure(figsize=(12,5))
    sns.violinplot(data=melted, x='PC', y='Val', hue='Label', split=True, inner='quartile')
    plt.title("Feature Distribution by Class"); plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_feature_correlation(X, save_path=None):
    Xr, n_components = safe_pca(X, 10)
    df = pd.DataFrame(Xr, columns=[f"PC{i+1}" for i in range(n_components)])
    corr = df.corr()
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("PCA Correlation Matrix"); plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_boxplot(X, y, save_path=None):
    Xr, n_components = safe_pca(X, 5)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(Xr, columns=cols)
    df['Label'] = y
    melted = df.melt(id_vars='Label', var_name='PC', value_name='Val')
    fig = plt.figure(figsize=(12,5))
    sns.boxplot(data=melted, x='PC', y='Val', hue='Label')
    plt.title("Boxplot of PCA Components"); plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_violinplot(X, y, save_path=None):
    Xr, n_components = safe_pca(X, 5)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(Xr, columns=cols)
    df['Label'] = y
    melted = df.melt(id_vars='Label', var_name='PC', value_name='Val')
    fig = plt.figure(figsize=(12,5))
    sns.violinplot(data=melted, x='PC', y='Val', hue='Label', split=True)
    plt.title("Violin Plot of PCA Components"); plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def plot_training_curves(history, out_dir="results", prefix="", wandb=None):
    print("Called plot_training_curves with history keys:", history.keys())
    os.makedirs(out_dir, exist_ok=True)
    metrics = ['loss', 'acc', 'f1', 'roc_auc']
    for metric in metrics:
        plt.figure()
        has_data = False
        train_key   = f"train_{metric}"
        val_key     = f"val_{metric}"
        val_bal_key = f"val_bal_{metric}"

        num_epochs = 0
        for k in [train_key, val_key, val_bal_key]:
            if k in history:
                num_epochs = max(num_epochs, len(history[k]))

        if train_key in history and len(history[train_key]) > 0:
            plt.plot(range(1, len(history[train_key]) + 1), history[train_key], label="train", linestyle='-')
            has_data = True
        if val_key in history and len(history[val_key]) > 0:
            plt.plot(range(1, len(history[val_key]) + 1), history[val_key], label="val", linestyle='--')
            has_data = True
        if val_bal_key in history and len(history[val_bal_key]) > 0:
            plt.plot(range(1, len(history[val_bal_key]) + 1), history[val_bal_key], label="val_bal", linestyle=':')
            has_data = True

        if has_data:
            plt.title(f"{metric.title()} Curve")
            plt.xlabel("Epoch")
            plt.ylabel(metric.title())
            plt.legend()
            plt.xticks(np.arange(1, num_epochs + 1, 1))   # integer epoch ticks
            path = os.path.join(out_dir, f"{prefix}{metric}_curve.png")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            if wandb:
                wandb.log({f"{prefix}{metric}_curve": wandb.Image(path)})
        else:
            plt.close()


def visualize_roc_compare(y_nat, p_nat, y_bal, p_bal, save_path=None):
    y_nat = np.asarray(y_nat).reshape(-1)
    p_nat = np.asarray(p_nat).reshape(-1)
    y_bal = np.asarray(y_bal).reshape(-1)
    p_bal = np.asarray(p_bal).reshape(-1)

    fpr_n, tpr_n, _ = roc_curve(y_nat, p_nat)
    fpr_b, tpr_b, _ = roc_curve(y_bal, p_bal)
    auc_n = auc(fpr_n, tpr_n)
    auc_b = auc(fpr_b, tpr_b)

    fig = plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr_n, tpr_n, label=f"Natural (AUC={auc_n:.3f})")
    plt.plot(fpr_b, tpr_b, label=f"Balanced (AUC={auc_b:.3f})")
    plt.plot([0,1],[0,1], 'k--', linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC: Natural vs Balanced")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def visualize_pr_compare(y_nat, p_nat, y_bal, p_bal, save_path=None):
    y_nat = np.asarray(y_nat).reshape(-1)
    p_nat = np.asarray(p_nat).reshape(-1)
    y_bal = np.asarray(y_bal).reshape(-1)
    p_bal = np.asarray(p_bal).reshape(-1)

    prec_n, rec_n, _ = precision_recall_curve(y_nat, p_nat)
    prec_b, rec_b, _ = precision_recall_curve(y_bal, p_bal)
    ap_n = average_precision_score(y_nat, p_nat)
    ap_b = average_precision_score(y_bal, p_bal)

    fig = plt.figure(figsize=(5.2, 4.2))
    plt.plot(rec_n, prec_n, label=f"Natural (AP={ap_n:.3f})")
    plt.plot(rec_b, prec_b, label=f"Balanced (AP={ap_b:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR: Natural vs Balanced")
    plt.legend(loc="lower left")
    plt.tight_layout()
    if save_path:
        _safe_save(fig, save_path)
    else:
        plt.show()

def plot_eval_overlays(y_nat, p_nat, y_bal, p_bal, out_dir="results", prefix="", wandb=None):
    os.makedirs(out_dir, exist_ok=True)
    roc_path = os.path.join(out_dir, f"{prefix}roc_overlay_nat_vs_bal.png")
    pr_path  = os.path.join(out_dir, f"{prefix}pr_overlay_nat_vs_bal.png")

    visualize_roc_compare(y_nat, p_nat, y_bal, p_bal, save_path=roc_path)
    visualize_pr_compare(y_nat, p_nat, y_bal, p_bal, save_path=pr_path)

    if wandb:
        wandb.log({
            f"{prefix}roc_overlay_nat_vs_bal": wandb.Image(roc_path),
            f"{prefix}pr_overlay_nat_vs_bal":  wandb.Image(pr_path),
        })


def plot_all_results(
        Xf, y, y_pred, y_prob, feature_groups,
        prefix="", plot_dir="results", wandb=None, plots_to_make=None, history=None
):
    os.makedirs(plot_dir, exist_ok=True)


    if plots_to_make is None:
        plots_to_make = [
            "confusion","roc","pr",
            "feature_distribution","feature_correlation",
            "feature_scatter","boxplot","violinplot","groupwise_pca"
        ]

    # -- confusion
    if "confusion" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}confusion_matrix.png")
        visualize_confusion_matrix(y, y_pred, save_path=p)
        if wandb:
              wandb.log({f"{prefix}confusion": wandb.Image(p)})

    # -- ROC
    if "roc" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}roc_curve.png")
        visualize_roc(y, y_prob, save_path=p)
        if wandb:
            wandb.log({f"{prefix}roc": wandb.Image(p)})

    # -- PR
    if "pr" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}pr_curve.png")
        visualize_precision_recall(y, y_prob, save_path=p)
        if wandb:
            wandb.log({f"{prefix}pr": wandb.Image(p)})

    # sample for speed
    n = len(y)
    idx = np.random.choice(n, min(n,1000), replace=False)
    Xs, ys = Xf[idx], y[idx]

    # -- feature dist
    if "feature_distribution" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_distribution.png")
        visualize_feature_distribution(Xs, ys, save_path=p)
        if wandb:
            wandb.log({f"{prefix}feat_dist": wandb.Image(p)})

    # -- correlation
    if "feature_correlation" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_correlation.png")
        visualize_feature_correlation(Xs, save_path=p)
        if wandb:
            wandb.log({f"{prefix}feat_corr": wandb.Image(p)})

    # -- scatter
    if "feature_scatter" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_scatter.png")
        visualize_feature_scatter(Xs, ys, save_path=p)
        if wandb:
            wandb.log({f"{prefix}feat_scatter": wandb.Image(p)})

    # -- boxplot
    if "boxplot" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_boxplot.png")
        visualize_boxplot(Xs, ys, save_path=p)
        if wandb:
            wandb.log({f"{prefix}boxplot": wandb.Image(p)})

    # -- violin
    if "violinplot" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_violinplot.png")
        visualize_violinplot(Xs, ys, save_path=p)
        if wandb:
            wandb.log({f"{prefix}violinplot": wandb.Image(p)})

    # -- groupwise PCA
    if "groupwise_pca" in plots_to_make:
        visualize_groupwise_pca(Xs, ys, feature_groups, out_dir=plot_dir)
        if wandb:
            for name in feature_groups:
                sp = os.path.join(plot_dir, f"pca_scatter_{name.lower()}.png")
                vp = os.path.join(plot_dir, f"pca_violin_{name.lower()}.png")
                wandb.log({f"{prefix}pca_scatter_{name}": wandb.Image(sp)})
                wandb.log({f"{prefix}pca_violin_{name}": wandb.Image(vp)})

