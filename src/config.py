import argparse
import torch


def get_train_config():
    """
    Get configuration arguments for training QuICPTMs.
    """
    parser = argparse.ArgumentParser(description='QuICPTMs: Mamba + DyGeo-GAT for PTM Site Prediction')

    # --- 1. Dataset and System Settings ---
    parser.add_argument('--Y', action='store_true', help='Select the Y dataset')
    parser.add_argument('--ST', action='store_true', help='Select the ST dataset')
    parser.add_argument('--SulBert', action='store_true', help='Select the SulBert dataset')
    parser.add_argument('--S2', action='store_true', help='Select the S2 dataset')

    parser.add_argument('--device', type=int, default=0,
                        choices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0],
                        help='GPU device index to use for computation')
    parser.add_argument('--path', type=str, default=None, help='Specific path for model evaluation')
    parser.add_argument('--learn-name', type=str, default='QuICPTMs_S_Site',
                        help='Name identifier for the training run')
    parser.add_argument('--save-best', type=bool, default=True, help='Whether to save the best model parameters')

    # --- 2. General Model Hyperparameters ---
    parser.add_argument('--max-len', type=int, default=1024,
                        help='Maximum sequence length (Must match the .npy file dimensions)')
    parser.add_argument('--dim-embedding', type=int, default=1280,
                        help='ESM embedding dimension (e.g., ESM-2 650M is 1280)')
    parser.add_argument('--num-class', type=int, default=2, help='Number of classification classes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    # --- 3. Architecture Specifics (Mamba & GNN) ---
    parser.add_argument('--gnn-hidden-dim', type=int, default=256, help='Hidden dimension for GNN')
    parser.add_argument('--gnn-out-dim', type=int, default=256, help='Output dimension for GNN')
    parser.add_argument('--gnn-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--gnn-heads', type=int, default=16, help='Number of attention heads for DyGeo-GAT')

    parser.add_argument('--mamba-d-state', type=int, default=32, help='Mamba parameter: d_state')
    parser.add_argument('--mamba-d-conv', type=int, default=4, help='Mamba parameter: d_conv')
    parser.add_argument('--mamba-expand', type=int, default=2, help='Mamba parameter: expand factor')

    parser.add_argument('--fusion-num-queries', type=int, default=64,
                        help='Number of learnable queries for the QDF-Net fusion module')

    # --- 4. Training Parameters ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--k-fold', type=int, default=5, help='Number of folds for cross-validation')

    # --- 5. Loss Function (DVIC) Hyperparameters ---
    parser.add_argument('--loss-alpha', type=float, default=1.0,
                        help='Weight for the Jensen-Shannon Divergence (Consistency) loss.')
    parser.add_argument('--loss-beta', type=float, default=0.1,
                        help='Weight for the Supervised Contrastive loss.')
    parser.add_argument('--loss-temperature', type=float, default=0.07,
                        help='Temperature scaling for the Contrastive loss.')

    config, _ = parser.parse_known_args()
    return config