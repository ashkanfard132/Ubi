import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, fbeta_score
import torch
import random
from itertools import product
try:
    from cuml.svm import SVC as CuMLSVC
    _HAS_CUML_SVM = True
except Exception:
    _HAS_CUML_SVM = False
    from sklearn.svm import SVC as SklearnSVC




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


def get_optimizer(model, optim_type='adam', lr=0.001, weight_decay=0.0):
    if optim_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'amsgrad':
        return optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")


def get_scheduler(optimizer, sched_type='step', step_size=10, gamma=0.1, t_max= 5):
    if sched_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'exp':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=3)
    elif sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
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



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     

def evaluate_batches(model, data, labels, batch_size=32, device='cpu', tokenizer_or_batch_converter=None, model_name=None, loss_fn=None):
    model.eval()
    all_outputs = []
    all_targets = []
    loss_total = 0.0

    with torch.no_grad():
        n = len(data)
        for i in range(0, n, batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.float32).to(device)

            if model_name == "prot_bert":
                tokenizer = tokenizer_or_batch_converter
                batch = tokenizer(
                    list(batch_data),
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                )
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            elif model_name == "esm2_t6_8m":
                batch_converter = tokenizer_or_batch_converter
                batch_seqs = [("protein", str(seq)) for seq in batch_data]
                _, _, batch_tokens = batch_converter(batch_seqs)
                batch_tokens = batch_tokens.to(device)
                outputs = model(tokens=batch_tokens)
            else:
                x_dtype = torch.long if model.__class__.__name__ != 'MLPClassifier' else torch.float32
                x_batch = torch.tensor(batch_data, dtype=x_dtype).to(device)
                outputs = model(x_batch)

            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if loss_fn is not None:
                loss = loss_fn(outputs, batch_labels)
                loss_total += float(loss.item()) * len(batch_labels)
            all_outputs.append(outputs.cpu())
            all_targets.append(batch_labels.cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    probs = torch.sigmoid(torch.from_numpy(all_outputs)).numpy()
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_targets, probs)
    except:
        roc_auc = float('nan')
    avg_loss = loss_total / len(all_targets) if loss_fn is not None else float('nan')

    return avg_loss, acc, prec, rec, f1, roc_auc

def protbert_space(seq: str) -> str:

    valid = set("ACDEFGHIKLMNPQRSTVWYBXZOU")  
    seq = ''.join((c if c.upper() in valid else 'X') for c in seq.upper())
    return ' '.join(list(seq))

def _parse_list(s, allow_strings=False):
    """
    Turn a comma-separated string (or list) into a list of values.
    If allow_strings=True, keep tokens like 'scale','auto','rbf', etc.
    Otherwise, try float conversion.
    """
    if isinstance(s, (list, tuple)):
        items = list(s)
    else:
        items = [x.strip() for x in str(s).split(",") if x.strip()]

    out = []
    for x in items:
        if allow_strings and x.lower() in {"scale", "auto", "rbf", "linear", "poly", "sigmoid"}:
            out.append(x.lower())
            continue
        try:
            out.append(float(x))
        except ValueError:
            if allow_strings:
                out.append(x)
          
    return out


from itertools import product
from sklearn.metrics import roc_auc_score, f1_score

def _svm_grid_search(train_X, train_y, val_X, val_y, args, wandb_run=None):
    """
    Very small, local grid search on the validation set to pick SVM params.
    Uses cuML SVC if available; otherwise sklearn SVC.
    Scoring metric: combination of ROC AUC and F1.
    """
    kernels = _parse_list(getattr(args, "svm_kernels", "rbf,linear"), allow_strings=True)
    C_vals  = [float(c) for c in _parse_list(getattr(args, "svm_C", "0.1,1,5,10"))]
    gammas_raw = _parse_list(getattr(args, "svm_gamma", "scale,auto,0.001,0.0001"), allow_strings=True)

    if _HAS_CUML_SVM:
        gamma_vals = [g for g in gammas_raw if isinstance(g, (int, float))]
        if not gamma_vals:
            gamma_vals = [1e-3, 1e-4]
    else:
        gamma_vals = gammas_raw

    combos = list(product(kernels, C_vals, gamma_vals)) or [("rbf", 1.0, 1e-3)]

    best_score = -1.0
    best_auc = -1.0
    best_f1 = -1.0
    best = None
    best_model = None

    print(f"[SVM grid] Trying {len(combos)} combinations...")

    for ker, C, gamma in combos:
        try:
            if _HAS_CUML_SVM:
                model = CuMLSVC(kernel=str(ker), C=float(C), gamma=float(gamma),
                                probability=True, verbose=False)
            else:
                model = SklearnSVC(kernel=str(ker), C=float(C), gamma=gamma,
                                   probability=True, class_weight="balanced",
                                   random_state=getattr(args, "seed", 42))

            model.fit(train_X, y=train_y)

            # Probabilities for AUC
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(val_X)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(val_X)
            else:
                scores = model.predict(val_X)

            preds = (scores >= 0.5).astype(int) if scores.ndim == 1 else model.predict(val_X)

            auc = roc_auc_score(val_y, scores)
            f1  = f1_score(val_y, preds)

       
            combined_score = auc + f1

            if wandb_run:
                wandb_run.log({
                    "svm_grid/val_auc": float(auc),
                    "svm_grid/val_f1": float(f1),
                    "svm_grid/combined_score": float(combined_score),
                    "svm_grid/kernel": str(ker),
                    "svm_grid/C": float(C),
                    "svm_grid/gamma": float(gamma) if isinstance(gamma, (int, float)) else str(gamma)
                })

            if combined_score > best_score:
                best_score = combined_score
                best_auc = auc
                best_f1 = f1
                best = (ker, C, gamma)
                best_model = model
                print(f"[SVM grid] New best score={combined_score:.4f} (AUC={auc:.4f}, F1={f1:.4f}) | "
                      f"kernel={ker} C={C} gamma={gamma}")

        except Exception as e:
            print(f"[SVM grid] Skipped kernel={ker} C={C} gamma={gamma} due to error: {e}")

    if best_model is None:
        print("[SVM grid] No valid combo fit; falling back to default SVC.")
        if _HAS_CUML_SVM:
            best_model = CuMLSVC(kernel="rbf", C=5.0, gamma=1e-4, probability=True, verbose=False)
        else:
            best_model = SklearnSVC(kernel="rbf", C=5.0, gamma="scale",
                                    probability=True, class_weight="balanced",
                                    random_state=getattr(args, "seed", 42))
        best_model.fit(train_X, y=train_y)
        best = ("rbf", 5.0, "scale")
        best_auc = float("nan")
        best_f1 = float("nan")

    setattr(best_model, "fit_already", True)

    print(f"[SVM grid] Best params: kernel={best[0]} C={best[1]} gamma={best[2]} "
          f"(val AUC={best_auc:.4f}, val F1={best_f1:.4f}, combined={best_score:.4f})")
    if wandb_run:
        wandb_run.log({
            "svm_grid/best_kernel": str(best[0]),
            "svm_grid/best_C": float(best[1]),
            "svm_grid/best_gamma": float(best[2]) if isinstance(best[2], (int, float)) else str(best[2]),
            "svm_grid/best_val_auc": float(best_auc),
            "svm_grid/best_val_f1": float(best_f1),
            "svm_grid/best_combined_score": float(best_score)
        })

    return best_model, {"kernel": best[0], "C": best[1], "gamma": best[2]}, best_score

