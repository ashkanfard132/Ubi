
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
    """Helper to save a figure, print errors, assert success, then close."""
    try:
        fig.savefig(path)
        assert os.path.isfile(path), f"File not found after save: {path}"
        # print(f"[✔] Saved plot to {path}")
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

        # Scatter
        fig = plt.figure(figsize=(6,5))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Label', alpha=0.6)
        plt.title(f'PCA Scatter: {group_name}')
        plt.tight_layout()
        path = os.path.join(out_dir, f"pca_scatter_{group_name.lower()}.png")
        _safe_save(fig, path)

        # Violin
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
    # Xr = PCA(n_components=2).fit_transform(X)
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
    # Xr = PCA(n_components=5).fit_transform(X)
    # cols = [f"PC{i+1}" for i in range(5)]
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
    # Xr = PCA(n_components=10).fit_transform(X)
    # df = pd.DataFrame(Xr, columns=[f"PC{i+1}" for i in range(10)])
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
    # Xr = PCA(n_components=5).fit_transform(X)
    # cols = [f"PC{i+1}" for i in range(5)]
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
    # Xr = PCA(n_components=5).fit_transform(X)
    # cols = [f"PC{i+1}" for i in range(5)]
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

def plot_all_results(
    Xf, y, y_pred, y_prob, feature_groups,
    prefix="", plot_dir="results", wandb_run=None, plots_to_make=None
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
        if wandb_run: wandb_run.log({f"{prefix}confusion": wandb_run.Image(p)})

    # -- ROC
    if "roc" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}roc_curve.png")
        visualize_roc(y, y_prob, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}roc": wandb_run.Image(p)})

    # -- PR
    if "pr" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}pr_curve.png")
        visualize_precision_recall(y, y_prob, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}pr": wandb_run.Image(p)})

    # sample for speed
    n = len(y)
    idx = np.random.choice(n, min(n,1000), replace=False)
    Xs, ys = Xf[idx], y[idx]

    # -- feature dist
    if "feature_distribution" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_distribution.png")
        visualize_feature_distribution(Xs, ys, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}feat_dist": wandb_run.Image(p)})

    # -- correlation
    if "feature_correlation" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_correlation.png")
        visualize_feature_correlation(Xs, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}feat_corr": wandb_run.Image(p)})

    # -- scatter
    if "feature_scatter" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_scatter.png")
        visualize_feature_scatter(Xs, ys, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}feat_scatter": wandb_run.Image(p)})

    # -- boxplot
    if "boxplot" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_boxplot.png")
        visualize_boxplot(Xs, ys, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}boxplot": wandb_run.Image(p)})

    # -- violin
    if "violinplot" in plots_to_make:
        p = os.path.join(plot_dir, f"{prefix}feature_violinplot.png")
        visualize_violinplot(Xs, ys, save_path=p)
        if wandb_run: wandb_run.log({f"{prefix}violinplot": wandb_run.Image(p)})

    # -- groupwise PCA
    if "groupwise_pca" in plots_to_make:
        visualize_groupwise_pca(Xs, ys, feature_groups, out_dir=plot_dir)
        if wandb_run:
            for name in feature_groups:
                sp = os.path.join(plot_dir, f"pca_scatter_{name.lower()}.png")
                vp = os.path.join(plot_dir, f"pca_violin_{name.lower()}.png")
                wandb_run.log({f"{prefix}pca_scatter_{name}": wandb_run.Image(sp)})
                wandb_run.log({f"{prefix}pca_violin_{name}": wandb_run.Image(vp)})
