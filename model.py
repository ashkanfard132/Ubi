import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from args import get_args

try:
    from cuml.svm import SVC as CuMLSVC
    HAS_CUMLSVM = True
except Exception as e:
    print(f"[INFO] cuML not available or incompatible GPU, falling back to scikit-learn SVC.\nReason: {e}")
    from sklearn.svm import SVC as SklearnSVC
    HAS_CUMLSVM = False

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import esm
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# --- Torch Models ---

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.5):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, num_filters=64, kernel_size=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = torch.max(x, dim=2).values
        x = self.dropout(x)
        return self.fc(x).squeeze(1)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_size=64, dropout=0.3, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[0], h_n[1]), dim=1)  # Concatenate forward/backward final states
        h = self.dropout(h)
        return self.fc(h).squeeze(1)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, nhead=8, num_layers=2, max_len=100, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def _generate_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)


# --- Pretrained Models Loader ---

def get_pretrained_model(model_name, device='cpu', freeze=True):
    """
    Load a pretrained protein model from HuggingFace or ESM.
    Returns model and its tokenizer/processor.
    """
    if model_name == 'prot_bert':
        
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, model_max_length=512)
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        return model.to(device), tokenizer
    elif model_name == 'esm2_t6_8m':
        
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        return model.to(device), batch_converter
    else:
        raise ValueError(f"Unknown pretrained model: {model_name}")

# --- Classification Head for Pretrained Encoders ---

class PretrainedClassifierHead(nn.Module):
    def __init__(self, encoder, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        if hasattr(encoder, 'config') and hasattr(encoder.config, 'hidden_size'):
            hidden_size = encoder.config.hidden_size  # ProtBERT
        elif hasattr(encoder, 'embed_dim'):
            hidden_size = encoder.embed_dim          # ESM
        else:
            hidden_size = 768  # fallback

        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # <-- Added
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids=None, attention_mask=None, tokens=None):
        # HuggingFace (ProtBERT)
        if hasattr(self.encoder, 'config'):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embed = outputs.last_hidden_state[:, 0, :]
        # ESM
        elif hasattr(self.encoder, 'embed_tokens'):
            results = self.encoder(tokens, repr_layers=[self.encoder.num_layers])
            token_representations = results["representations"][self.encoder.num_layers]
            cls_embed = token_representations[:, 0, :]
        else:
            raise ValueError("Unsupported encoder type for PretrainedClassifierHead")
        x = self.relu(self.fc1(cls_embed))
        x = self.dropout(x)  # <-- Added
        return self.fc2(x).squeeze(1)


# --- Factory Functions ---

def get_torch_model(name, input_dim_or_vocab=None, device='cpu', freeze=True, dropout=0.5):
    name = name.lower()
    if name == 'mlp':
        return MLPClassifier(input_dim=input_dim_or_vocab, dropout=dropout)
    elif name == 'cnn':
        return CNNClassifier(vocab_size=input_dim_or_vocab, dropout=dropout)
    elif name == 'lstm':
        return LSTMClassifier(vocab_size=input_dim_or_vocab, dropout=dropout)
    elif name == 'transformer':
        return TransformerClassifier(vocab_size=input_dim_or_vocab, dropout=dropout)
    elif name in ['prot_bert', 'esm2_t6_8m']:
        encoder, tokenizer_or_batch = get_pretrained_model(name, device=device, freeze=freeze)
        # >>> Expanded section: add dropout to classifier head <<<
        classifier = PretrainedClassifierHead(encoder, dropout=dropout)
        return classifier, tokenizer_or_batch
    else:
        raise ValueError(f"Unknown torch model: {name}")

def get_ml_model(name, y=None, random_state=42):
    """
    Returns an sklearn-compatible classifier tuned for
    imbalanced biological site-prediction tasks.
    
    """
    name = name.lower()
    
    # Precompute counts if labels provided
    if y is not None:
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
    else:
        n_pos = n_neg = 0

    if name == 'rf':
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

    elif name == 'svm':
        if HAS_CUMLSVM:
            use_probability = True 
            if use_probability:
                return CuMLSVC(
                    kernel='rbf',
                    C=5.0,
                    probability=True,
                    gamma= 0.0005,
                    verbose=False
                    # DO NOT set class_weight!
                )
            else:
                return CuMLSVC(
                    kernel='rbf',
                    class_weight='balanced',
                    C=5.0,
                    probability=False,
                    gamma='scale',
                    verbose=False
                )
        else:
            return SklearnSVC(
                probability=True,
                class_weight='balanced',
                kernel='rbf',
                C=5.0,
                gamma='scale',
                random_state=random_state
            )


    elif name == 'xgb':
       
        if n_pos > 0:
            scale_pos_weight = float(n_neg) / float(n_pos)
        else:
            scale_pos_weight = 1.0

        return XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            n_estimators=800,
            learning_rate=0.01,
            max_depth=6,
            gamma=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            random_state=random_state,
            n_jobs=-1
        )

    elif name in ('cat', 'catboost'):
       
        if n_pos > 0:
            total = n_pos + n_neg
            class_weights = [n_neg/total, n_pos/total]
        else:
            class_weights = [1.0, 1.0]

        return CatBoostClassifier(
            iterations=1500,
            depth=7,
            learning_rate=0.01,
            auto_class_weights='Balanced', 
            l2_leaf_reg=5,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )

    elif name in ('ada', 'adaboost'):
        base = DecisionTreeClassifier(
            max_depth=1,
            class_weight='balanced',
            random_state=random_state
        )
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=150,
            learning_rate=0.7,
            random_state=random_state
        )


    elif name in ('logistic', 'lreg', 'logreg'):
        return LogisticRegression(
            class_weight='balanced',
            solver='saga',
            max_iter=500,
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown ML model: {name}")

