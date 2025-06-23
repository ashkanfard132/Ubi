import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

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
      
