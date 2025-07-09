import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Ubiquitination Site Prediction")
    parser.add_argument('--model', type=str, required=True,
                        choices=[
                            'mlp', 'cnn', 'lstm', 'transformer',
                            'rf', 'xgb', 'ada', 'cat', 'svm','lreg',
                            'prot_bert', 'esm2_t6_8m'
                        ],
                        help='Model type: mlp, cnn, lstm, transformer, rf, xgb, ada, cat, svm, prot_bert, esm2_t6_8m')
    parser.add_argument('--fasta', type=str, required=True, help='Path to first FASTA file')
    parser.add_argument('--excel', type=str, required=True, help='Path to first Excel file')
    parser.add_argument('--fasta2', type=str, help='Path to second FASTA file')
    parser.add_argument('--excel2', type=str, help='Path to second Excel file')
    parser.add_argument('--pssm1', type=str, help='Path to PSSM folder for plant 1')
    parser.add_argument('--pssm2', type=str, help='Path to PSSM folder for plant 2')
    parser.add_argument('--features', nargs='+', default=['aac', 'dpc', 'tpc'],
                        help='List of features to use')
    parser.add_argument('--window_size', type=int, default=21,
                        help='Sliding window size (odd number between 5 and 35)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'mse', 'mae', 'focal'])
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight for BCE loss. If not set, will use n_neg/n_pos')
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop', 'adamw', 'amsgrad'])
    parser.add_argument('--sched', type=str, default='step',
                        choices=['step', 'exp', 'plateau', 'cosine', 'none'])
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--t_max', type=int, default=10,
                        help='T_max for cosine annealing scheduler')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--wandb_project', type=str, default='ubiquitination-prediction',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, required=True, help='WandB entity name')
    parser.add_argument('--plots', nargs='+',
                        choices=['distribution', 'confusion', 'roc', 'pr'],
                        help='Select plots to display: distribution, confusion, roc, pr')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='WandB API Key (optional, for notebook login)')
    parser.add_argument('--sampling', nargs='+', type=str, default=['none'],
                        choices=['over', 'under', 'smote', 'smotee', 'none'],
                        help="Apply one or more: over, under, smote, smotee, none (in order) to TRAIN set")
    parser.add_argument('--sampling_ratios', nargs='+', type=float, default=[1.0],
                        help="Ratio(s) for sampling, e.g. --sampling_ratios 0.5 1.0 (must match --sampling)")
    parser.add_argument('--freeze_pretrained', action='store_true',
                        help="If set, do NOT finetune pretrained encoder weights (feature extraction mode)")
    parser.add_argument('--cv', type=int, default=1,
                        help="Number of cross-validation folds (e.g. 10 for 10-fold CV; 1 = no CV, just standard split)")
    parser.add_argument('--dropout', type=float, default=0.5,
        help='Dropout probability (0 disables dropout, typical: 0.3-0.5 for MLP/CNN, 0.2-0.3 for LSTM, 0.1-0.3 for Transformer)'
    )
    parser.add_argument('--best_threshold', action='store_true',
    help='Find best threshold from validation set and use for test (ML models only)')

    return parser.parse_args()
