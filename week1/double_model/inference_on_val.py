import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import PrecomputedNPZDataset
from model import ClassifierModel, QuantileRegressionModel
from utils import get_device, QuantileLoss
import json
from tqdm import tqdm
import os

# Config
BATCH_SIZE = 64
DEVICE = get_device()
VAL_FILE = "data/val.npz"
VOCAB_FILE = "data/train_vocab.json"
CLASSIFIER_PATH = 'pth'
REGRESSOR_PATH = 'pth'

# Quantiles used during training
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

# Load vocab sizes
with open(VOCAB_FILE, 'r') as f:
    vocabs = json.load(f)

domain_vocab_size = len(vocabs['domain_vocab'])
tld_vocab_size = len(vocabs['tld_vocab'])
user_vocab_size = len(vocabs['user_vocab'])

# Load val dataset (regression mode to get proper targets)
val_dataset = PrecomputedNPZDataset(VAL_FILE, task="regression")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get input dims
sample_batch = val_dataset[0]
features_num_sample, title_emb_sample, *_ = sample_batch
vector_size_title = title_emb_sample.shape[0]
vector_size_num = features_num_sample.shape[0]

# Load models
classifier = ClassifierModel(
    vector_size_num=vector_size_num,
    vector_size_title=vector_size_title,
    scale=3,
    domain_vocab_size=domain_vocab_size,
    tld_vocab_size=tld_vocab_size,
    user_vocab_size=user_vocab_size
).to(DEVICE)

regressor = QuantileRegressionModel(
    vector_size_num=vector_size_num,
    vector_size_title=vector_size_title,
    scale=3,
    domain_vocab_size=domain_vocab_size,
    tld_vocab_size=tld_vocab_size,
    user_vocab_size=user_vocab_size,
    num_quantiles=len(quantiles)
).to(DEVICE)

# Load your saved models here:
classifier_ckpt = torch.load(CLASSIFIER_PATH)
regressor_ckpt = torch.load(REGRESSOR_PATH)

classifier.load_state_dict(classifier_ckpt['model_state_dict'])
regressor.load_state_dict(regressor_ckpt['model_state_dict'])

classifier.eval()
regressor.eval()

# Inference loop
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="Inference"):
        features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]

        logits = classifier(features_num, title_emb, domain_idx, tld_idx, user_idx).squeeze()
        probs = torch.sigmoid(logits)

        is_nonzero = (probs > 0.5).float()

        # Initialize prediction vector
        batch_preds = torch.zeros_like(target)

        # If classifier predicts non-zero → run regressor
        if is_nonzero.sum() > 0:
            nonzero_mask = (is_nonzero == 1)

            features_num_nz = features_num[nonzero_mask]
            title_emb_nz = title_emb[nonzero_mask]
            domain_idx_nz = domain_idx[nonzero_mask]
            tld_idx_nz = tld_idx[nonzero_mask]
            user_idx_nz = user_idx[nonzero_mask]

            reg_output = regressor(features_num_nz, title_emb_nz, domain_idx_nz, tld_idx_nz, user_idx_nz)
            median_preds = reg_output[:, 2]  # 50th percentile

            batch_preds[nonzero_mask] = median_preds

        all_preds.append(batch_preds.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# Stack full predictions
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# MAE in log scale
mae_log = np.mean(np.abs(all_preds - all_targets))
print(f"MAE (log space): {mae_log:.4f}")

# If you want real-scale MAE (inverse log10 transform):
real_preds = 10**all_preds - 1
real_targets = 10**all_targets - 1
mae_real = np.mean(np.abs(real_preds - real_targets))
print(f"MAE (real scale): {mae_real:.4f}")
