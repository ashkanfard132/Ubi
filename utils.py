import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, fbeta_score

def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = float('nan')
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "Sp": specificity,
        "MCC": matthews_corrcoef(y_true, y_pred)
    }


def get_loss(loss_type='bce', pos_weight=None):
    if loss_type == 'bce':
        if pos_weight is not None:
            # pos_weight should be a scalar tensor: torch.tensor(float)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'focal':
        def focal_loss(logits, targets, alpha=1, gamma=2):
            bce = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
            pt = torch.exp(-bce)
            focal = alpha * (1 - pt) ** gamma * bce
            return focal.mean()
        return focal_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def get_optimizer(model, optim_type='adam', lr=0.001):
    if optim_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optim_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optim_type == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optim_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    elif optim_type == 'amsgrad':
        return optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

def get_scheduler(optimizer, sched_type='step', step_size=10, gamma=0.1):
    if sched_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'exp':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=3)
    elif sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif sched_type == 'none':
        return None 
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def make_balanced_test(Xf_test, Xs_test, y_test, random_state=42):
    np.random.seed(random_state)
    idx_pos = np.where(y_test == 1)[0]
    idx_neg = np.where(y_test == 0)[0]
    n = min(len(idx_pos), len(idx_neg))
    idx_pos_sample = np.random.choice(idx_pos, n, replace=False)
    idx_neg_sample = np.random.choice(idx_neg, n, replace=False)
    idx_balanced = np.concatenate([idx_pos_sample, idx_neg_sample])
    np.random.shuffle(idx_balanced)
    return Xf_test[idx_balanced], Xs_test[idx_balanced], y_test[idx_balanced]


def find_best_threshold(y_true, y_probs, metric='f1'):
    best_thresh = 0.5
    best_score = -np.inf
    thresholds = np.linspace(0.01, 0.99, 100)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        if metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score

