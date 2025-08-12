import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, load_embeddings, get_device
from dataloader import PrecomputedNPZDataset
from model import FullModel
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import json
from torchinfo import summary

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = get_device()
TRAIN_FILE = "data/train.npz"
VAL_FILE = "data/val.npz"
VOCAB_FILE = "data/train_vocab.json"

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'models/{timestamp}'

    # Load vocab sizes from vocab file
    with open(VOCAB_FILE, 'r') as f:
        vocabs = json.load(f)

    # Dataset
    train_dataset = PrecomputedNPZDataset(TRAIN_FILE, time_decay=None)
    val_dataset = PrecomputedNPZDataset(VAL_FILE, time_decay=None)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    sample_batch = train_dataset[0]
    features_num_sample, title_emb_sample, *_ = sample_batch
    # Save config used to build the model
    config = {
        'vector_size_title': title_emb_sample.shape[0],
        'vector_size_num': features_num_sample.shape[0],
        'scale': 3,
        'domain_vocab_size': len(vocabs['domain_vocab']),
        'tld_vocab_size': len(vocabs['tld_vocab']),
        'user_vocab_size' : len(vocabs['user_vocab'])
    }

    model = FullModel(**config).to(DEVICE)
    criterion = nn.MSELoss()

    # Load back the model from 'best_model_4.pth'
    checkpoint_path = 'models/20250613_134308/best_model_5.pth'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # --- Permutation Feature Importance ---
    print("\nCalculating permutation feature importance on validation set...")

    # Reload validation data as a single batch for simplicity
    val_features_num = []
    val_title_emb = []
    val_domain_idx = []
    val_tld_idx = []
    val_user_idx = []
    val_targets = []

    for batch in val_dataloader:
        features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.cpu() for b in batch]
        val_features_num.append(features_num)
        val_title_emb.append(title_emb)
        val_domain_idx.append(domain_idx)
        val_tld_idx.append(tld_idx)
        val_user_idx.append(user_idx)
        val_targets.append(target)

    features_num = torch.cat(val_features_num, dim=0).to(DEVICE)
    title_emb = torch.cat(val_title_emb, dim=0).to(DEVICE)
    domain_idx = torch.cat(val_domain_idx, dim=0).to(DEVICE)
    tld_idx = torch.cat(val_tld_idx, dim=0).to(DEVICE)
    user_idx = torch.cat(val_user_idx, dim=0).to(DEVICE)
    targets = torch.cat(val_targets, dim=0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        base_output = model(features_num, title_emb, domain_idx, tld_idx, user_idx)
        base_loss = criterion(base_output, targets.unsqueeze(1)).mean().item()
        base_mae = torch.mean(torch.abs(base_output.squeeze() - targets)).item()

    print(f"Base Val Loss: {base_loss:.4f}, Base MAE: {base_mae:.4f}")

    import copy

    feature_importances = []
    for i in range(features_num.shape[1]):
        features_num_permuted = features_num.clone()
        idx = torch.randperm(features_num_permuted.shape[0])
        features_num_permuted[:, i] = features_num_permuted[idx, i]
        with torch.no_grad():
            output = model(features_num_permuted, title_emb, domain_idx, tld_idx, user_idx)
            loss = criterion(output, targets.unsqueeze(1)).mean().item()
            mae = torch.mean(torch.abs(output.squeeze() - targets)).item()
        importance = loss - base_loss
        feature_importances.append((f"num_{i}", importance, mae - base_mae))

    # Permute title embedding as a whole
    idx = torch.randperm(title_emb.shape[0])
    title_emb_permuted = title_emb[idx]
    with torch.no_grad():
        output = model(features_num, title_emb_permuted, domain_idx, tld_idx, user_idx)
        loss = criterion(output, targets.unsqueeze(1)).mean().item()
        mae = torch.mean(torch.abs(output.squeeze() - targets)).item()
    feature_importances.append(("title_emb", loss - base_loss, mae - base_mae))

    # Permute domain_idx
    idx = torch.randperm(domain_idx.shape[0])
    domain_idx_permuted = domain_idx[idx]
    with torch.no_grad():
        output = model(features_num, title_emb, domain_idx_permuted, tld_idx, user_idx)
        loss = criterion(output, targets.unsqueeze(1)).mean().item()
        mae = torch.mean(torch.abs(output.squeeze() - targets)).item()
    feature_importances.append(("domain_idx", loss - base_loss, mae - base_mae))

    # Permute tld_idx
    idx = torch.randperm(tld_idx.shape[0])
    tld_idx_permuted = tld_idx[idx]
    with torch.no_grad():
        output = model(features_num, title_emb, domain_idx, tld_idx_permuted, user_idx)
        loss = criterion(output, targets.unsqueeze(1)).mean().item()
        mae = torch.mean(torch.abs(output.squeeze() - targets)).item()
    feature_importances.append(("tld_idx", loss - base_loss, mae - base_mae))

    # Permute user_idx
    idx = torch.randperm(user_idx.shape[0])
    user_idx_permuted = user_idx[idx]
    with torch.no_grad():
        output = model(features_num, title_emb, domain_idx, tld_idx, user_idx_permuted)
        loss = criterion(output, targets.unsqueeze(1)).mean().item()
        mae = torch.mean(torch.abs(output.squeeze() - targets)).item()
    feature_importances.append(("user_idx", loss - base_loss, mae - base_mae))

    # Print importances sorted by loss increase
    print("\nPermutation Feature Importances (all features):")
    for idx, imp, mae_imp in sorted(feature_importances, key=lambda x: -x[1]):
        print(f"Feature {idx}: ΔLoss={imp:.4f}, ΔMAE={mae_imp:.4f}")

