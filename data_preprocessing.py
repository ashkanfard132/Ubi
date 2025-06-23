import numpy as np
import pandas as pd
from collections import defaultdict
import os
from sklearn.model_selection import StratifiedKFold
import os
import re

AMINO_ACIDS = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]

# Amino acid index mapping (0: padding/'X', 1-20: standard amino acids)
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['X'] = 0  # Padding/unknown

def kfold_split(Xf, Xs, y, n_splits=10, random_state=42):
    """
    Generator that yields splits for cross-validation.
    Each split yields (train_idx, val_idx, test_idx).
    Test set is one fold, validation is one fold from the remaining, rest is train.
    """
    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(y))
    for fold, (trainval_idx, test_idx) in enumerate(skf_outer.split(Xf, y)):
        # Now, from trainval, select val fold
        Xf_trainval, Xs_trainval, y_trainval = Xf[trainval_idx], Xs[trainval_idx], y[trainval_idx]
        skf_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1000 + fold)
        inner_split = skf_inner.split(Xf_trainval, y_trainval)
        train_idx_rel, val_idx_rel = next(inner_split)  # Take first as val fold
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]
        yield {
            "fold": fold + 1,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx
        }

def encode_sequence_windows(seq_windows, encoding='int'):
    """
    Encode sequence windows for deep learning models.
    Args:
        seq_windows: np.array, shape (n_samples, window_size), each entry is a single-letter AA code.
        encoding: 'int' for integer encoding, 'onehot' for one-hot (for CNN, LSTM, transformer)
    Returns:
        np.array: shape (n_samples, window_size) for 'int', or (n_samples, window_size, 21) for 'onehot'
    """
    seq_windows = np.asarray(seq_windows)
    if encoding == 'onehot':
        one_hot = np.zeros((seq_windows.shape[0], seq_windows.shape[1], len(AMINO_ACIDS) + 1), dtype=np.float32)
        for i, window in enumerate(seq_windows):
            for j, aa in enumerate(window):
                idx = AA_TO_IDX.get(aa, 0)
                one_hot[i, j, idx] = 1.0
        return one_hot
    else:  # 'int'
        return np.array([[AA_TO_IDX.get(aa, 0) for aa in window] for window in seq_windows], dtype=np.int64)


def oversample(X, y, ratio=1.0, random_state=42, return_indices=False):
    np.random.seed(random_state)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = len(idx_pos)

    desired_n_pos = int(n_pos * ratio)
    if desired_n_pos <= n_pos:
        idx_pos_sample = np.random.choice(idx_pos, size=desired_n_pos, replace=False)
    else:
        idx_pos_sample = np.random.choice(idx_pos, size=desired_n_pos, replace=True)
    idx_sample = np.concatenate([idx_neg, idx_pos_sample])
    np.random.shuffle(idx_sample)
    if return_indices:
        return idx_sample
    return X[idx_sample], y[idx_sample]

def undersample(X, y, ratio=1.0, random_state=42, return_indices=False):
    np.random.seed(random_state)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_neg = len(idx_neg)

    if ratio <= 0:
        print("[WARN] Ratio <= 0; skipping undersampling.")
        idx_sample = np.arange(len(y))
    else:
        desired_n_neg = int(n_neg * ratio)
        desired_n_neg = min(desired_n_neg, n_neg)
        idx_neg_sample = np.random.choice(idx_neg, size=desired_n_neg, replace=False)
        idx_sample = np.concatenate([idx_pos, idx_neg_sample])
        np.random.shuffle(idx_sample)
    if return_indices:
        return idx_sample
    return X[idx_sample], y[idx_sample]


def smote_sample(X, y, ratio=1.0, random_state=42):
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(sampling_strategy=ratio, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def smotee_sample(X, y, ratio=1.0, random_state=42):
    from imblearn.combine import SMOTEENN
    smoteenn = SMOTEENN(sampling_strategy=ratio, random_state=random_state)
    X_res, y_res = smoteenn.fit_resample(X, y)
    return X_res, y_res


VALID_FEATURES = {'aac', 'dpc', 'tpc', 'pssm', 'physicochem', 'binary', 'be'}

def read_fasta_ordered(fasta_path):
    """Reads a FASTA file and returns a list of (sequence_id, sequence) tuples, in order."""
    proteins = []
    seq_id, seq = None, ''
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    proteins.append((seq_id, seq))
                header = line[1:].split()[0].strip()
                seq_id = header.split("|")[1] if "|" in header else header
                seq = ''
            else:
                seq += line
        if seq_id is not None:
            proteins.append((seq_id, seq))
    return proteins

def pssm_id_to_file_map_by_order(fasta_path, pssm_folder):
    """
    Maps proteins by order in FASTA to the files in the folder (sorted numerically!).
    Handles files named '1.pssm', '(1).pssm', etc.
    Returns {protein_id: filepath}
    """
    def numeric_part(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    
    proteins = read_fasta_ordered(fasta_path)
    pssm_files = sorted(
        [f for f in os.listdir(pssm_folder) if f.endswith('.pssm')],
        key=numeric_part
    )

    if len(pssm_files) < len(proteins):
        print(f"[WARN] Fewer PSSM files ({len(pssm_files)}) than proteins ({len(proteins)}).")
    mapping = {}
    for i, (prot_id, _) in enumerate(proteins):
        if i < len(pssm_files):
            mapping[prot_id] = os.path.join(pssm_folder, pssm_files[i])
        else:
            print(f"[WARN] No PSSM file for protein {prot_id}")
    return mapping

def parse_pssm_folder_by_order(fasta_path, pssm_folder):
    """
    Returns {protein_id: pssm_matrix} dict using mapping by order.
    Only the first 20 scoring columns are kept.
    """
    id2file = pssm_id_to_file_map_by_order(fasta_path, pssm_folder)
    pssm_data = {}
    for prot_id, filepath in id2file.items():
        try:
            with open(filepath) as f:
                lines = f.readlines()
            data_lines = [line for line in lines if line.strip() and line.strip()[0].isdigit()]
            matrix = []
            for line in data_lines:
                fields = line.strip().split()
                if len(fields) >= 22:
                    matrix.append([int(x) for x in fields[2:22]])
            pssm_data[prot_id] = np.array(matrix)
        except Exception as e:
            print(f"[WARN] Could not parse PSSM for {prot_id}: {e}")
            pssm_data[prot_id] = None
            
    return pssm_data



def read_fasta(fasta_path):
    """Reads a FASTA file and returns a dictionary of {sequence_id: sequence}."""
    sequences = {}
    seq_id = None
    seq = ''
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    sequences[seq_id] = seq
                header = line[1:].split()[0].strip()
                seq_id = header.split("|")[1] if "|" in header else header
                seq = ''
            else:
                seq += line
        if seq_id is not None:
            sequences[seq_id] = seq
    return sequences

# def parse_pssm_folder(pssm_folder):
#     """Parses a folder of PSSM .csv files into a dictionary {protein_id: np.array}."""
#     if not pssm_folder or not os.path.exists(pssm_folder):
#         return None
#     pssm_data = {}
#     for file in os.listdir(pssm_folder):
#         if file.endswith(".csv"):
#             pid = os.path.splitext(file)[0]
#             df = pd.read_csv(os.path.join(pssm_folder, file), header=None)
#             pssm_data[pid] = df.values
#     return pssm_data

def load_dataset(
    fasta_path,
    excel_path,
    selected_features=None,
    pssm_data=None,
    window_size=21
):
    """Loads dataset, extracts features, and prepares labels for ML models."""

    if selected_features is None:
        selected_features = ['aac', 'dpc', 'tpc']

    # Validate features
    unknown = set(selected_features) - VALID_FEATURES
    if unknown:
        raise ValueError(f"Unknown features: {unknown}. Supported: {VALID_FEATURES}")

    sequences = read_fasta(fasta_path)
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    # Clean Protein ID and Position columns
    df['Protein ID'] = df['Protein ID'].astype(str).str.strip()
    df = df[df['Protein ID'].notna() & df['Position'].notna()]

    # Build positives dictionary: {protein_id: set(positions)}
    positives = defaultdict(set)
    for _, row in df.iterrows():
        try:
            positives[row['Protein ID']].add(int(row['Position']) - 1)
        except Exception:
            continue

    # Diagnostics: Check for missing IDs and out-of-bound positions
    total_positives = sum(len(v) for v in positives.values())
    missing_prot_id = 0
    missing_pos = 0
    for prot_id, positions in positives.items():
        if prot_id not in sequences:
            print(f"[WARN] Protein ID '{prot_id}' in Excel not found in FASTA!")
            missing_prot_id += len(positions)
        else:
            seq_len = len(sequences[prot_id])
            for pos in positions:
                if pos < 0 or pos >= seq_len:
                    print(f"[WARN] Out-of-bounds: '{prot_id}' position {pos+1} (protein length {seq_len})")
                    missing_pos += 1
    print(f"Total positive sites in Excel: {total_positives}")
    print(f"Not matched (Protein ID not in FASTA): {missing_prot_id}")
    print(f"Skipped (positions out of sequence): {missing_pos}")

    X_feat, X_seq, y = [], [], []
    skipped = 0
    feature_groups = {}
    offset = 0

    # Physicochemical property lookup tables
    physicochemical_props = {
        "hydrophobicity": {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        },
        "charge": {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
            'Q': 0, 'E': -1, 'G': 0, 'H': 0.5, 'I': 0,
            'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
            'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
        },
        "polarity": {
            'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
            'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2,
            'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0,
            'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9
        }
    }
    physicochem_keys = list(physicochemical_props.keys())

    for prot_id, seq in sequences.items():
        seq_len = len(seq)
        prot_pssm = pssm_data[prot_id] if (pssm_data and prot_id in pssm_data) else None

        for pos in range(seq_len):
            label = 1 if (prot_id in positives and pos in positives[prot_id]) else 0

            # Window boundaries
            half_window = window_size // 2
            start = max(0, pos - half_window)
            end = min(seq_len, pos + half_window + 1)
            pad_left = half_window - (pos - start) if end - start < window_size else 0
            pad_right = half_window - (end - pos - 1) if end - start < window_size else 0

            window_seq = ['X'] * pad_left + list(seq[start:end]) + ['X'] * pad_right
            assert len(window_seq) == window_size

            features = []

            # 1. AAC (Amino Acid Composition)
            if 'aac' in selected_features:
                aac = [window_seq.count(aa) / window_size for aa in AMINO_ACIDS]
                features.extend(aac)
                if 'AAC' not in feature_groups:
                    feature_groups['AAC'] = list(range(offset, offset + len(aac)))
                    offset += len(aac)

            # 2. DPC (Dipeptide Composition)
            if 'dpc' in selected_features:
                dpc = [
                    sum(1 for i in range(window_size - 1)
                        if window_seq[i] == aa1 and window_seq[i + 1] == aa2) / (window_size - 1)
                    for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS
                ]
                features.extend(dpc)
                if 'DPC' not in feature_groups:
                    feature_groups['DPC'] = list(range(offset, offset + len(dpc)))
                    offset += len(dpc)

            # 3. TPC (Tripeptide Composition)
            if 'tpc' in selected_features:
                tpc = [
                    sum(1 for i in range(window_size - 2)
                        if window_seq[i] == aa1 and window_seq[i + 1] == aa2 and window_seq[i + 2] == aa3) / (window_size - 2)
                    for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS for aa3 in AMINO_ACIDS
                ]
                features.extend(tpc)
                if 'TPC' not in feature_groups:
                    feature_groups['TPC'] = list(range(offset, offset + len(tpc)))
                    offset += len(tpc)

            # 4. PSSM
            # if 'pssm' in selected_features and prot_pssm is not None:
            #     pssm_feat = []
            #     for i in range(start, end):
            #         if i < 0 or i >= prot_pssm.shape[0]:
            #             pssm_feat.extend([0] * 20)
            #         else:
            #             pssm_feat.extend(list(prot_pssm[i]))
            #     pssm_feat = [0] * 20 * pad_left + pssm_feat + [0] * 20 * pad_right
            #     features.extend(pssm_feat)
            #     if 'PSSM' not in feature_groups:
            #         feature_groups['PSSM'] = list(range(offset, offset + len(pssm_feat)))
            #         offset += len(pssm_feat)
            # Assume: window_seq is always length window_size
            if 'pssm' in selected_features and prot_pssm is not None:
                # For position `pos`, the PSSM window should match the window_seq (which is already padded with Xs).
                pssm_window = []
                for aa_idx, aa in enumerate(window_seq):
                    pssm_row = [0] * 20  # Default for padding/unknown
                    seq_idx = pos - (window_size // 2) + aa_idx
                    if 0 <= seq_idx < prot_pssm.shape[0] and aa != 'X':
                        pssm_row = list(prot_pssm[seq_idx])
                    pssm_window.extend(pssm_row)
                features.extend(pssm_window)
                
                if 'PSSM' not in feature_groups:
                    feature_groups['PSSM'] = list(range(offset, offset + len(pssm_window)))
                    offset += len(pssm_window)


            # 5. Physicochemical properties (average per window, per property)
            if 'physicochem' in selected_features:
                props = [
                    np.mean([physicochemical_props[prop].get(aa, 0) for aa in window_seq])
                    for prop in physicochem_keys
                ]
                features.extend(props)
                if 'Physicochem' not in feature_groups:
                    feature_groups['Physicochem'] = list(range(offset, offset + len(props)))
                    offset += len(props)

            # 6. Binary-Encoding (BE): 21 x 20 = 420 dims
            if 'binary' in selected_features or 'be' in selected_features:
                binary = []
                for aa in window_seq:
                    binary.extend([1 if aa == a else 0 for a in AMINO_ACIDS])
                features.extend(binary)
                if 'Binary' not in feature_groups:
                    feature_groups['Binary'] = list(range(offset, offset + len(binary)))
                    offset += len(binary)

            # Collect features, sequences, and labels
            if features:
                X_feat.append(features)
                X_seq.append(window_seq)
                y.append(label)
            else:
                skipped += 1

    X_feat_np = np.array(X_feat)
    y_np = np.array(y)
    print(f"[INFO] Loaded {len(y_np)} samples: {sum(y_np == 1)} positives, {sum(y_np == 0)} negatives. Skipped {skipped} entries.")

    return X_feat_np, np.array(X_seq), y_np, feature_groups

