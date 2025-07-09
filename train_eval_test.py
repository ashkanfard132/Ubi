import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_preprocessing import (
    encode_sequence_windows,
    oversample,
    undersample,
    smote_sample,
    smotee_sample,
    kfold_split
)
from utils import get_loss, get_optimizer, get_scheduler, make_balanced_test, compute_metrics, find_best_threshold
from tqdm import tqdm

def train_and_evaluate(
    args,
    Xf, Xs, y, feature_groups,
    get_torch_model, get_ml_model, PretrainedClassifierHead,
    encode_sequence_windows,
    make_balanced_test,
    oversample, undersample, smote_sample, smotee_sample,
    get_loss, get_optimizer, get_scheduler,
    train_model, evaluate_model
):
    """
    Handles both standard split and k-fold cross-validation,
    including all sampling, model training, and evaluation.
    Returns: (metrics_nat, metrics_bal)
    """

    SEQ_MODELS = ['cnn', 'lstm', 'transformer', 'prot_bert', 'esm2_t6_8m']
    FEAT_MODELS = ['mlp', 'rf', 'xgb', 'ada', 'cat', 'svm', 'lreg']

    metrics_nat_list = []
    metrics_bal_list = []

    if args.cv > 1:
        
        splits = kfold_split(Xf, Xs, y, n_splits=args.cv)
        fold_metrics_nat = []
        fold_metrics_bal = []
        fold_metrics_val_bal = []
        fold_best_thresholds = []
        last_out = None
        for split in splits:
            fold_num = split.get("fold", len(fold_metrics_nat)+1)
            print(f"\n======== Fold {fold_num} / {args.cv} ========")
            train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
            Xf_train, Xf_val, Xf_test = Xf[train_idx], Xf[val_idx], Xf[test_idx]
            Xs_train, Xs_val, Xs_test = Xs[train_idx], Xs[val_idx], Xs[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
            Xf_val_bal, Xs_val_bal, y_val_bal = make_balanced_test(Xf_val, Xs_val, y_val)
            Xf_test_bal, Xs_test_bal, y_test_bal = make_balanced_test(Xf_test, Xs_test, y_test)
            
            # Do all sampling, training, evaluation, etc. as below:
            out = _one_fold_train_eval(
                args, Xf_train, Xs_train, y_train,
                Xf_val, Xs_val, y_val,
                Xf_test, Xs_test, y_test,
                Xf_val_bal, Xs_val_bal, y_val_bal,
                Xf_test_bal, Xs_test_bal, y_test_bal,
                get_torch_model, get_ml_model, PretrainedClassifierHead,
                encode_sequence_windows,
                oversample, undersample, smote_sample, smotee_sample,
                get_loss, get_optimizer, get_scheduler,
                train_model, evaluate_model,
                SEQ_MODELS, FEAT_MODELS
            )
            last_out = out
            fold_metrics_nat.append(out["metrics_nat"])
            fold_metrics_bal.append(out["metrics_bal"])
            # Only append val_bal for DL
            if out.get("metrics_val_bal", None) is not None:
                fold_metrics_val_bal.append(out["metrics_val_bal"])

            if "best_thresh" in out:   # <-- NEW
                fold_best_thresholds.append(out["best_thresh"])

            print(f"Fold {fold_num} NATURAL metrics:")
            for k, v in out["metrics_nat"].items():
                print(f"  {k}: {v:.4f}")
            print(f"Fold {fold_num} BALANCED metrics:")
            for k, v in out["metrics_bal"].items():
                print(f"  {k}: {v:.4f}")
            # if out.get("metrics_val_bal", None) is not None:
            #     print(f"Fold {fold_num} VAL BALANCED (DL only):")
            #     for k, v in out["metrics_val_bal"].items():
            #         print(f"  {k}: {v:.4f}")
        
        # Compute mean (final) metrics
        metrics_nat = {k: np.mean([m[k] for m in fold_metrics_nat]) for k in fold_metrics_nat[0]}
        metrics_bal = {k: np.mean([m[k] for m in fold_metrics_bal]) for k in fold_metrics_bal[0]}
        if fold_metrics_val_bal:
            metrics_val_bal = {k: np.mean([m[k] for m in fold_metrics_val_bal]) for k in fold_metrics_val_bal[0]}
        else:
            metrics_val_bal = None

        final_best_thresh = float(np.mean(fold_best_thresholds)) if fold_best_thresholds else 0.5

        print("\n========== Final Mean Metrics (NATURAL TEST) ==========")
        for k, v in metrics_nat.items():
            print(f"{k}: {v:.4f}")
        print("\n========== Final Mean Metrics (BALANCED TEST) ==========")
        for k, v in metrics_bal.items():
            print(f"{k}: {v:.4f}")
        if metrics_val_bal is not None:
            print("\n========== Final Mean Metrics (VAL BALANCED, DL only) ==========")
            for k, v in metrics_val_bal.items():
                print(f"{k}: {v:.4f}")

        return {
              "metrics_nat":     metrics_nat,
              "metrics_bal":     metrics_bal,
              "metrics_val_bal": metrics_val_bal,
              "final_best_thresh": final_best_thresh,
              "y_test":          last_out["y_test"],
              "y_pred_nat":      last_out["y_pred_nat"],
              "y_prob_nat":      last_out["y_prob_nat"],
              "y_test_bal":      last_out["y_test_bal"],
              "y_pred_bal":      last_out["y_pred_bal"],
              "y_prob_bal":      last_out["y_prob_bal"],
              
              "Xf_test":         Xf_test,
              "Xf_test_bal":     Xf_test_bal,
          }

    else:
        
        Xf_trainval, Xf_test, Xs_trainval, Xs_test, y_trainval, y_test = train_test_split(
            Xf, Xs, y, test_size=0.2, stratify=y, random_state=42
        )
        Xf_train, Xf_val, Xs_train, Xs_val, y_train, y_val = train_test_split(
            Xf_trainval, Xs_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=42
        )
        Xf_val_bal, Xs_val_bal, y_val_bal = make_balanced_test(Xf_val, Xs_val, y_val)
        Xf_test_bal, Xs_test_bal, y_test_bal = make_balanced_test(Xf_test, Xs_test, y_test)

        out = _one_fold_train_eval(
            args, Xf_train, Xs_train, y_train,
            Xf_val, Xs_val, y_val,
            Xf_test, Xs_test, y_test,
            Xf_val_bal, Xs_val_bal, y_val_bal,
            Xf_test_bal, Xs_test_bal, y_test_bal,
            get_torch_model, get_ml_model, PretrainedClassifierHead,
            encode_sequence_windows,
            oversample, undersample, smote_sample, smotee_sample,
            get_loss, get_optimizer, get_scheduler,
            train_model, evaluate_model,
            SEQ_MODELS, FEAT_MODELS
        )
        return dict(
            metrics_nat=out["metrics_nat"],
            metrics_bal=out["metrics_bal"],
            metrics_val_bal=out.get("metrics_val_bal", None),
            final_best_thresh=out.get("best_thresh", 0.5),
            y_test=out["y_test"],
            y_pred_nat=out["y_pred_nat"],
            y_prob_nat=out["y_prob_nat"],
            y_test_bal=out["y_test_bal"],
            y_pred_bal=out["y_pred_bal"],
            y_prob_bal=out["y_prob_bal"],
            Xf_test=Xf_test,                
            Xf_test_bal=Xf_test_bal,        
            # optionally, for sequences:
            Xs_test=Xs_test,
            Xs_test_bal=Xs_test_bal,
        )



def _one_fold_train_eval(
    args,
    Xf_train, Xs_train, y_train,
    Xf_val, Xs_val, y_val,
    Xf_test, Xs_test, y_test,
    Xf_val_bal, Xs_val_bal, y_val_bal,
    Xf_test_bal, Xs_test_bal, y_test_bal,
    get_torch_model, get_ml_model, PretrainedClassifierHead,
    encode_sequence_windows,
    oversample, undersample, smote_sample, smotee_sample,
    get_loss, get_optimizer, get_scheduler,
    train_model, evaluate_model,
    SEQ_MODELS, FEAT_MODELS
):
    model_name = args.model.lower()
    # --- Sampling ---
    sampling_steps = args.sampling
    sampling_ratios = args.sampling_ratios
    if len(sampling_ratios) == 1 and len(sampling_steps) > 1:
        sampling_ratios = sampling_ratios * len(sampling_steps)
    assert len(sampling_steps) == len(sampling_ratios), "Number of --sampling and --sampling_ratios must match!"

    for method, ratio in zip(sampling_steps, sampling_ratios):
        method = method.lower()
        if method == 'over':
            idx_sample = oversample(Xf_train, y_train, ratio=ratio, random_state=42, return_indices=True)
            Xf_train = Xf_train[idx_sample]
            Xs_train = Xs_train[idx_sample]
            y_train = y_train[idx_sample]
        elif method == 'under':
            idx_sample = undersample(Xf_train, y_train, ratio=ratio, random_state=42, return_indices=True)
            Xf_train = Xf_train[idx_sample]
            Xs_train = Xs_train[idx_sample]
            y_train = y_train[idx_sample]
        elif method == 'smote':
            Xf_train, y_train = smote_sample(Xf_train, y_train, ratio=ratio, random_state=42)
        elif method == 'smotee':
            Xf_train, y_train = smotee_sample(Xf_train, y_train, ratio=ratio, random_state=42)
        elif method == 'none':
            continue
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
    num_positives = np.sum(y_train)
    num_negatives = len(y_train) - num_positives
    print(f"After sampling, Positives: {num_positives}, Negatives: {num_negatives}")
    # --- Prepare data for model ---
    if args.model.lower() in FEAT_MODELS:
        # --- Feature Scaling: Standardize features for ML models only! ---

        scaler = StandardScaler()
        Xf_train = scaler.fit_transform(Xf_train)
        Xf_val = scaler.transform(Xf_val)
        Xf_val_bal = scaler.transform(Xf_val_bal)
        Xf_test = scaler.transform(Xf_test)
        Xf_test_bal = scaler.transform(Xf_test_bal)
        # print("Mean/std of train PSSM after scaling:", np.mean(Xf_train), np.std(Xf_train))

        train_data, val_data = Xf_train, Xf_val
        val_data_bal = Xf_val_bal
        val_labels_bal = y_val_bal 
        test_data_natural = Xf_test
        test_data_balanced = Xf_test_bal

    elif args.model.lower() in SEQ_MODELS:
        if len(Xs_train) != len(Xf_train):
            Xs_train = Xs_train[:len(Xf_train)]
        def seq_to_aa_string(seq):
            if isinstance(seq, str):
                return seq
            elif isinstance(seq, np.ndarray):
                # If it's an array of single characters
                if seq.dtype.kind in {'U', 'S'}:  # unicode or bytes
                    return ''.join(seq.tolist())
                # If it's an array of ints, you must have a mapping!
                # e.g. [1,2,3] --> "ARN"
                else:
                    raise ValueError("You are passing integer-encoded sequences to ESM2, which expects AA strings.")
            else:
                raise ValueError(f"Unknown sequence type: {type(seq)}")

        if model_name in ['prot_bert', 'esm2_t6_8m']:
            train_data = [seq_to_aa_string(seq) for seq in Xs_train]
            val_data = [seq_to_aa_string(seq) for seq in Xs_val]
            val_data_bal = [seq_to_aa_string(seq) for seq in Xs_val_bal]
            test_data_natural = [seq_to_aa_string(seq) for seq in Xs_test]
            test_data_balanced = [seq_to_aa_string(seq) for seq in Xs_test_bal]

        else:
            train_data = encode_sequence_windows(Xs_train, encoding='int')
            val_data = encode_sequence_windows(Xs_val, encoding='int')
            val_data_bal = encode_sequence_windows(Xs_val_bal, encoding='int')
            test_data_natural = encode_sequence_windows(Xs_test, encoding='int')
            test_data_balanced = encode_sequence_windows(Xs_test_bal, encoding='int')
        val_labels_bal = y_val_bal

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # --- Model ---
    # model_name = args.model.lower()
    if model_name in ['mlp', 'cnn', 'lstm', 'transformer']:
        input_dim = Xf_train.shape[1] if model_name == 'mlp' else 21
        model = get_torch_model(model_name, input_dim, dropout=args.dropout, device=args.device)
        use_torch = True
        tokenizer_or_batch_converter = None
    elif model_name in ['prot_bert', 'esm2_t6_8m']:
        model, tokenizer_or_batch_converter = get_torch_model(
            model_name, device=args.device, freeze=args.freeze_pretrained, dropout=args.dropout
        )
        use_torch = True
    else:
        model = get_ml_model(model_name, y=y_train)
        use_torch = False
        tokenizer_or_batch_converter = None


    # --- Training and evaluation ---
    if use_torch:
        import torch
        device = torch.device(args.device)
        pos_weight = None
        if args.loss == "bce" and args.pos_weight is not None:
            pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32).to(device)
        loss_fn = get_loss(args.loss, pos_weight=pos_weight)
        optimizer = get_optimizer(model, args.optim, args.lr)
        if args.sched == "cosine":
            scheduler = get_scheduler(optimizer, args.sched, t_max=args.t_max)
        else:
            scheduler = get_scheduler(optimizer, args.sched, args.step_size, args.gamma)

        model = train_model(args,
            model, train_data, y_train, epochs=args.epochs, batch_size=args.batch_size,
            device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
            val_data=val_data, val_labels=y_val,
            val_data_bal=val_data_bal, val_labels_bal=val_labels_bal,
            tokenizer_or_batch_converter=tokenizer_or_batch_converter,    
            model_name=model_name                                         
        )

        metrics_nat, y_pred_nat, y_prob_nat = evaluate_model(
            model, test_data_natural, y_test, device=device, 
            tokenizer_or_batch_converter=tokenizer_or_batch_converter, 
            model_name=model_name
        )
        metrics_bal, y_pred_bal, y_prob_bal = evaluate_model(
            model, test_data_balanced, y_test_bal, device=device, 
            tokenizer_or_batch_converter=tokenizer_or_batch_converter, 
            model_name=model_name
        )
        metrics_val_bal, y_pred_val_bal, y_prob_val_bal = evaluate_model(
            model, val_data_bal, y_val_bal, device=device, 
            tokenizer_or_batch_converter=tokenizer_or_batch_converter, 
            model_name=model_name
        )
    else:
        model.fit(train_data, y_train)

        # --- Compute best threshold on validation set (F2) if requested ---
        if getattr(args, "best_threshold", False) and hasattr(model, "predict_proba"):
            y_val_probs = model.predict_proba(val_data)[:, 1]
            best_thresh, best_f2 = find_best_threshold(y_val, y_val_probs, metric='f2')
            print(f"[INFO] Best threshold on val set (F2): {best_thresh:.3f} (F2={best_f2:.4f})")
        else:
            best_thresh = 0.5  # Default threshold

        # --- Test predictions using best threshold ---
        if hasattr(model, "predict_proba"):
            y_prob_nat = model.predict_proba(test_data_natural)[:, 1]
            y_prob_bal = model.predict_proba(test_data_balanced)[:, 1]
        else:
            y_prob_nat = model.predict(test_data_natural)
            y_prob_bal = model.predict(test_data_balanced)

        y_pred_nat = (y_prob_nat >= best_thresh).astype(int)
        y_pred_bal = (y_prob_bal >= best_thresh).astype(int)

        metrics_nat = compute_metrics(y_test, y_pred_nat, y_prob_nat)
        metrics_bal = compute_metrics(y_test_bal, y_pred_bal, y_prob_bal)

        metrics_val_bal = None


    return {
        "metrics_nat": metrics_nat,
        "metrics_bal": metrics_bal,
        "metrics_val_bal": metrics_val_bal,
        "y_test": y_test,
        "y_pred_nat": y_pred_nat,
        "y_prob_nat": y_prob_nat,
        "y_test_bal": y_test_bal,
        "y_pred_bal": y_pred_bal,
        "y_prob_bal": y_prob_bal,
        "best_thresh": best_thresh,
    }



def train_model(args,
    model,
    train_data,
    train_labels,
    epochs=10,
    batch_size=32,
    device='cpu',
    loss_fn=None,
    optimizer=None,
    scheduler=None,
    val_data=None,
    val_labels=None,
    val_data_bal=None,
    val_labels_bal=None,
    tokenizer_or_batch_converter=None,   
    model_name=None                     
):
    model.to(device)
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    dataset_size = len(train_data)
    for epoch in range(epochs):
        # Always shuffle after sampling
        perm = np.random.permutation(dataset_size)
        # Support both numpy array and list for train_data (especially for prot_bert or esm!)
        if isinstance(train_data, np.ndarray):
            train_data_shuf = train_data[perm]
        else:
            train_data_shuf = [train_data[i] for i in perm]
        if isinstance(train_labels, np.ndarray):
            train_labels_shuf = train_labels[perm]
        else:
            train_labels_shuf = [train_labels[i] for i in perm]


        all_outputs = []
        all_targets = []

        for start_idx in tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch+1}"):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_labels = torch.tensor(train_labels_shuf[start_idx:end_idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            if model_name == "prot_bert":
                tokenizer = tokenizer_or_batch_converter
                batch_seqs = list(train_data_shuf[start_idx:end_idx])
                batch = tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=args.window_size
                )
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, batch_labels)
            elif model_name == "esm2_t6_8m":
                batch_converter = tokenizer_or_batch_converter
                # Force all sequences to str
                batch_seqs = [("protein", str(seq)) for seq in train_data_shuf[start_idx:end_idx]]
                _, _, batch_tokens = batch_converter(batch_seqs)
                batch_tokens = batch_tokens.to(device)
                outputs = model(tokens=batch_tokens)
                loss = loss_fn(outputs, batch_labels)

            else:
                x_dtype = torch.long if model.__class__.__name__ != 'MLPClassifier' else torch.float32
                x_batch = torch.tensor(train_data_shuf[start_idx:end_idx], dtype=x_dtype).to(device)
                outputs = model(x_batch)
                loss = loss_fn(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(batch_labels.detach().cpu())


        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        probs = 1 / (1 + np.exp(-all_outputs))
        preds = (probs >= 0.5).astype(int)

        accuracy = accuracy_score(all_targets, preds)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        f1 = f1_score(all_targets, preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_targets, probs)
        except ValueError:
            roc_auc = float('nan')

        # === VALIDATION ===

        if val_data is not None and val_labels is not None:
            model.eval()
            with torch.no_grad():
                # --- CHANGED: handle prot_bert batching ---
                if model_name == "prot_bert":
                    tokenizer = tokenizer_or_batch_converter
                    val_seqs = list(val_data)
                    batch = tokenizer(
                        val_seqs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True
                    )
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs_val = model(input_ids=input_ids, attention_mask=attention_mask)
                    y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)
                elif model_name == "esm2_t6_8m":
                    batch_converter = tokenizer_or_batch_converter
                    val_data_tuples = [("protein", seq) for seq in val_data]
                    _, _, batch_tokens = batch_converter(val_data_tuples)
                    batch_tokens = batch_tokens.to(device)
                    outputs_val = model(tokens=batch_tokens)
                    y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)
                else:
                    x_dtype = torch.long if model.__class__.__name__ != 'MLPClassifier' else torch.float32
                    x_val = torch.tensor(val_data, dtype=x_dtype).to(device)
                    outputs_val = model(x_val)
                    y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)
                loss_val = loss_fn(outputs_val, y_val)
                probs_val = torch.sigmoid(outputs_val).cpu().numpy()
                preds_val = (probs_val >= 0.5).astype(int)
                y_val_np = y_val.cpu().numpy()
                acc_val = accuracy_score(y_val_np, preds_val)
                prec_val = precision_score(y_val_np, preds_val, zero_division=0)
                rec_val = recall_score(y_val_np, preds_val, zero_division=0)
                f1_val = f1_score(y_val_np, preds_val, zero_division=0)
                try:
                    roc_auc_val = roc_auc_score(y_val_np, probs_val)
                except ValueError:
                    roc_auc_val = float('nan')

                print_str = (
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: Loss={loss.item():.4f} Acc={accuracy:.4f} Prec={precision:.4f} "
                    f"Rec={recall:.4f} F1={f1:.4f} ROC_AUC={roc_auc:.4f}\n"
                    f"  Val (natural): Loss={loss_val.item():.4f} Acc={acc_val:.4f} Prec={prec_val:.4f} "
                    f"Rec={rec_val:.4f} F1={f1_val:.4f} ROC_AUC={roc_auc_val:.4f}"
                )

                # BALANCED VAL, IF PROVIDED
                if val_data_bal is not None and val_labels_bal is not None:
                    # --- CHANGED: handle prot_bert batching ---
                    if model_name == "prot_bert":
                        batch_bal = tokenizer(
                            list(val_data_bal),
                            return_tensors='pt',
                            padding=True,
                            truncation=True
                        )
                        input_ids_bal = batch_bal["input_ids"].to(device)
                        attention_mask_bal = batch_bal["attention_mask"].to(device)
                        outputs_val_bal = model(input_ids=input_ids_bal, attention_mask=attention_mask_bal)
                        y_val_bal = torch.tensor(val_labels_bal, dtype=torch.float32).to(device)
                    elif model_name == "esm2_t6_8m":
                        batch_converter = tokenizer_or_batch_converter
                        val_bal_tuples = [("protein", seq) for seq in val_data_bal]
                        _, _, batch_tokens_bal = batch_converter(val_bal_tuples)
                        batch_tokens_bal = batch_tokens_bal.to(device)
                        outputs_val_bal = model(tokens=batch_tokens_bal)
                        y_val_bal = torch.tensor(val_labels_bal, dtype=torch.float32).to(device)
                    else:
                        x_dtype = torch.long if model.__class__.__name__ != 'MLPClassifier' else torch.float32
                        x_val_bal = torch.tensor(val_data_bal, dtype=x_dtype).to(device)
                        outputs_val_bal = model(x_val_bal)
                        y_val_bal = torch.tensor(val_labels_bal, dtype=torch.float32).to(device)
                    loss_val_bal = loss_fn(outputs_val_bal, y_val_bal)
                    probs_val_bal = torch.sigmoid(outputs_val_bal).cpu().numpy()
                    preds_val_bal = (probs_val_bal >= 0.5).astype(int)
                    y_val_bal_np = y_val_bal.cpu().numpy()
                    acc_val_bal = accuracy_score(y_val_bal_np, preds_val_bal)
                    prec_val_bal = precision_score(y_val_bal_np, preds_val_bal, zero_division=0)
                    rec_val_bal = recall_score(y_val_bal_np, preds_val_bal, zero_division=0)
                    f1_val_bal = f1_score(y_val_bal_np, preds_val_bal, zero_division=0)
                    try:
                        roc_auc_val_bal = roc_auc_score(y_val_bal_np, probs_val_bal)
                    except ValueError:
                        roc_auc_val_bal = float('nan')

                    print_str += (
                        f"\n  Val (balanced): Loss={loss_val_bal.item():.4f} Acc={acc_val_bal:.4f} "
                        f"Prec={prec_val_bal:.4f} Rec={rec_val_bal:.4f} "
                        f"F1={f1_val_bal:.4f} ROC_AUC={roc_auc_val_bal:.4f}"
                    )

                print(print_str)
            model.train()
        else:
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} | "
                f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | "
                f"F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}"
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.item())
            else:
                scheduler.step()

    return model



def evaluate_model(
    model, 
    test_data, 
    test_labels, 
    threshold=0.5, 
    device='cpu',
    tokenizer_or_batch_converter=None,   
    model_name=None                     
):
    """
    Evaluates a model. Supports MLP, CNN, LSTM, transformer, prot_bert (HuggingFace), esm2_t6_8m (ESM2).

    """
    model.eval()
    with torch.no_grad():
        # === HuggingFace ProtBERT ===
        if model_name == "prot_bert":
            tokenizer = tokenizer_or_batch_converter
            batch = tokenizer(
                list(test_data), 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()
        
        # === ESM2 (Fair-esm) ===
        elif model_name == "esm2_t6_8m":
            batch_converter = tokenizer_or_batch_converter
            # test_data should be list of strings (already AA seq)
            data = [("protein", seq) for seq in test_data]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            outputs = model(tokens=batch_tokens).cpu().numpy()
        
        # === Standard torch models ===
        else:
            x_dtype = torch.long if model.__class__.__name__ != 'MLPClassifier' else torch.float32
            x_test = torch.tensor(test_data, dtype=x_dtype).to(device)
            outputs = model(x_test).cpu().numpy()
        
        # === Post-processing ===
        probs = 1 / (1 + np.exp(-outputs))
        preds = (probs >= threshold).astype(int)
        
    metrics = compute_metrics(test_labels, preds, probs)


    return metrics, preds, probs


def predict(
    model, 
    data, 
    threshold=0.5, 
    device='cpu', 
    tokenizer_or_batch_converter=None,    
    model_name=None                     
):
    """
    Supports: MLP, CNN, LSTM, transformer, prot_bert, esm2_t6_8m.
    """
    model.eval()
    with torch.no_grad():
        # === ProtBERT (HuggingFace) ===
        if model_name == "prot_bert":
            tokenizer = tokenizer_or_batch_converter
            batch = tokenizer(
                list(data), 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()
        
        # === ESM2 ===
        elif model_name == "esm2_t6_8m":
            batch_converter = tokenizer_or_batch_converter
            data_list = [("protein", seq) for seq in data]
            _, _, batch_tokens = batch_converter(data_list)
            batch_tokens = batch_tokens.to(device)
            outputs = model(tokens=batch_tokens).cpu().numpy()
        
        # === Standard torch models ===
        else:
            x_dtype = torch.long if model.__class__.__name__ != 'MLPClassifier' else torch.float32
            x = torch.tensor(data, dtype=x_dtype).to(device)
            outputs = model(x).cpu().numpy()

        # --- Post-processing ---
        probs = 1 / (1 + np.exp(-outputs))
        preds = (probs >= threshold).astype(int)
    return preds, probs

