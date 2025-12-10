import os
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

# Project imports
from quicptms_model import QuICPTMs
from config import get_train_config
from ml_set import *  # Ensure this module is included in the repo

# Reproducibility and Performance
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class DVICLoss(nn.Module):
    """
    Dual-View Information-Constrained (DVIC) Loss.
    Combines Cross-Entropy, Jensen-Shannon Divergence, and Supervised Contrastive Loss.
    """

    def __init__(self, alpha: float, beta: float, temperature: float = 0.07):
        super(DVICLoss, self).__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("Loss weights alpha and beta must be non-negative.")

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.classification_loss = nn.CrossEntropyLoss()

    def _calculate_jsd_loss(self, logits_seq, logits_struct):
        p_seq = F.softmax(logits_seq, dim=1).clamp(min=1e-7, max=1 - 1e-7)
        p_struct = F.softmax(logits_struct, dim=1).clamp(min=1e-7, max=1 - 1e-7)
        m = 0.5 * (p_seq + p_struct)

        kl_p_m = F.kl_div(m.log(), p_seq, reduction='batchmean')
        kl_q_m = F.kl_div(m.log(), p_struct, reduction='batchmean')
        return 0.5 * (kl_p_m + kl_q_m)

    def _calculate_scl_loss(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(embeddings.device), 0
        )
        mask = mask * logits_mask

        anchor_dot_contrast = torch.div(torch.matmul(embeddings, embeddings.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-7)
        return -mean_log_prob_pos.mean()

    def forward(self, model_output, labels):
        logits_seq = model_output['logits_seq']
        logits_struct = model_output['logits_struct']
        logits_fused = model_output['logits_fused']
        fused_embeddings = model_output['fused_embeddings']

        loss_cls = self.classification_loss(logits_fused, labels)

        loss_consistency = torch.tensor(0.0, device=logits_fused.device)
        if self.alpha > 0:
            loss_consistency = self._calculate_jsd_loss(logits_seq, logits_struct)

        loss_scl = torch.tensor(0.0, device=logits_fused.device)
        if self.beta > 0:
            if len(torch.unique(labels)) > 1 and labels.numel() > 1:
                loss_scl = self._calculate_scl_loss(fused_embeddings, labels)

        total_loss = loss_cls + self.alpha * loss_consistency + self.beta * loss_scl

        return total_loss, {
            'total_loss': total_loss.item(),
            'cls_loss': loss_cls.item(),
            'jsd_loss': loss_consistency.item(),
            'scl_loss': loss_scl.item()
        }


def evaluate_model(test_embedding, test_str_nodes, test_labels, model):
    model.eval()
    with torch.no_grad():
        model_outputs = model(test_embedding, test_str_nodes)
        logits = model_outputs['logits_fused']
        probs = F.softmax(logits, dim=1)

        probs_valid = probs[:, 1]

        if test_labels.shape[0] == 0:
            return 0.0, 0.5, 0.5, probs

        try:
            auc = roc_auc_score(test_labels.cpu().numpy(), probs_valid.cpu().numpy())
            auprc = average_precision_score(test_labels.cpu().numpy(), probs_valid.cpu().numpy())
        except ValueError:
            auc, auprc = 0.5, 0.5

        predicted = torch.max(probs, 1)[1]
        correct = (predicted == test_labels).sum().item()
        acc = 100 * correct / test_labels.shape[0]

    return acc, auc, auprc, probs


def create_dataloader(embedding, str_nodes, labels, batch_size):
    dataset = TensorDataset(embedding.cpu(), str_nodes.cpu(), labels.cpu())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)


def save_checkpoint(model_state, fold, auc, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = f'fold{fold}_QuICPTMs_model.pt'
    save_path = os.path.join(save_dir, filename)
    torch.save(model_state, save_path, _use_new_zipfile_serialization=False)
    print(f'Model saved: {save_path} | AUC: {auc:.3f}\n')


class WarmupScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


def train_one_fold(fold, model, device, epochs, criterion, optimizer,
                   train_loader, val_embedding, val_str_nodes, val_labels,
                   scheduler, parameters, warmup_steps, args, patience):
    scaler = GradScaler(device=device)
    max_auc = 0
    epochs_no_improve = 0
    reduce_lr_triggered = False

    val_embedding = val_embedding.to(device)
    val_str_nodes = val_str_nodes.to(device)
    val_labels = val_labels.to(device)

    save_dir = './model/QuICPTMs_train'  # Default save path
    if args.Y:
        save_dir = './model/Y_train'
    elif args.ST:
        save_dir = './model/ST_train'
    elif args.SulBert:
        save_dir = './model/SulBert_train'
    elif args.S2:
        save_dir = './model/S2_train'

    for epoch in range(epochs):
        model.train()
        epoch_losses = collections.defaultdict(float)

        for embedding, str_nodes, labels in train_loader:
            embedding, str_nodes, labels = embedding.to(device, non_blocking=True), str_nodes.to(device,
                                                                                                 non_blocking=True), labels.to(
                device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(device_type=device.type):
                outputs = model(embedding, str_nodes)
                loss, components = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k, v in components.items():
                epoch_losses[k] += v

        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}

        # Validation
        val_acc, val_auc, val_auprc, _ = evaluate_model(val_embedding, val_str_nodes, val_labels, model)

        # Checkpointing
        if val_auc > max_auc and epoch > 10:
            max_auc = val_auc
            epochs_no_improve = 0
            save_checkpoint(model.state_dict(), fold, val_auc, save_dir)

            log_msg = f"Epoch {epoch + 1}: Saved Best Model | Total Loss {avg_losses['total_loss']:.3f} | Val AUC {val_auc:.3f}\n"
            print(log_msg.strip())
            with open(os.path.join(save_dir, "training_log.txt"), "a") as f:
                f.write(log_msg)
        else:
            if epoch > 10:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Best Val AUC: {max_auc:.4f}")
            break

        # Scheduler Logic
        if epoch < warmup_steps:
            if not reduce_lr_triggered:
                scheduler.step()
        elif epoch == warmup_steps:
            if not reduce_lr_triggered:
                print("Warmup finished. Switching to ReduceLROnPlateau.")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=3, verbose=True, threshold=1e-4
                )
                reduce_lr_triggered = True

        if reduce_lr_triggered:
            scheduler.step(val_auc)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch + 1} | Loss: {avg_losses['total_loss']:.3f} | Val ACC: {val_acc:.3f} | Val AUC: {val_auc:.3f}")


def run_cross_validation(parameters, x_train_emb, x_train_str, train_labels,
                         device, args, k_fold=5):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    criterion = DVICLoss(alpha=parameters.loss_alpha, beta=parameters.loss_beta,
                         temperature=parameters.loss_temperature).to(device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_emb, train_labels.cpu().numpy())):
        print(f"\n{'=' * 10} Fold {fold + 1} Processing {'=' * 10}")

        model = QuICPTMs(parameters).to(device, non_blocking=True)

        train_loader = create_dataloader(
            x_train_emb[train_idx], x_train_str[train_idx], train_labels[train_idx], batch_size=64
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=5)

        train_one_fold(
            fold, model, device, epochs=150, criterion=criterion, optimizer=optimizer,
            train_loader=train_loader,
            val_embedding=x_train_emb[val_idx], val_str_nodes=x_train_str[val_idx], val_labels=train_labels[val_idx],
            scheduler=warmup_scheduler, parameters=parameters, warmup_steps=5, args=args, patience=50
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuICPTMs for PTM Prediction")
    parser.add_argument('--Y', action='store_true', help='Use Y dataset')
    parser.add_argument('--ST', action='store_true', help='Use ST dataset')
    parser.add_argument('--SulBert', action='store_true', help='Use SulBert dataset')
    parser.add_argument('--S2', action='store_true', help='Use S2 dataset')
    parser.add_argument('--device', type=int, default=0, help='GPU device index')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    parameters = get_train_config()

    # Dataset Loading (Simplified logic for clarity)
    if args.ST:
        train_df, _ = data_read()
        x_emb, _ = embedding_load()
        x_str, _ = embedding_str_load()
    elif args.Y:
        train_df, _ = data_readY()
        x_emb, _ = embedding_loadY()
        x_str, _ = embedding_str_loadY()
    elif args.SulBert:
        train_df, _ = data_readSulBert()
        x_emb, _ = embedding_loadSulBert()
        x_str, _ = embedding_str_loadSulBert()
    elif args.S2:
        print("Loading S2 dataset...")
        train_df, _ = data_readS2()
        x_emb, _ = embedding_loadS2()
        emb_dir = '../embedding'
        x_str = torch.from_numpy(np.load(os.path.join(emb_dir, 'S2_train_str_embedding_nodes.npy')))
    else:
        raise ValueError("Please specify a dataset (e.g., --S2)")

    train_labels = torch.tensor(train_df.iloc[:, 0].values, dtype=torch.int64).cpu()
    x_emb = x_emb.cpu()
    x_str = x_str.cpu()

    print(f"Training Data: Embeddings {x_emb.shape} | Structure {x_str.shape} | Labels {train_labels.shape}")

    # Set seed for reproducibility
    torch.manual_seed(142)

    run_cross_validation(parameters, x_emb, x_str, train_labels, device, args)