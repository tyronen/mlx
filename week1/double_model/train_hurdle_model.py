import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_device, QuantileLoss
from dataloader import PrecomputedNPZDataset
from model import ClassifierModel, QuantileRegressionModel
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from torchinfo import summary

# Hyperparameters
BATCH_SIZE = 64
EPOCHS_CLASS = 5
EPOCHS_REG = 5
LEARNING_RATE_CLASS = 1e-4
LEARNING_RATE_REG = 1e-4
DEVICE = get_device()

TRAIN_FILE = "data/train.npz"
VAL_FILE = "data/val.npz"
VOCAB_FILE = "data/train_vocab.json"

if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'models/{timestamp}'
    log_dir = f'runs/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    with open(VOCAB_FILE, 'r') as f:
        vocabs = json.load(f)


    ### ---------- CLASSIFIER ----------
    train_class_dataset = PrecomputedNPZDataset(TRAIN_FILE, task="classification")
    val_class_dataset = PrecomputedNPZDataset(VAL_FILE, task="classification")

    train_class_dataloader = DataLoader(train_class_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_class_dataloader = DataLoader(val_class_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_batch = train_class_dataset[0]
    features_num_sample, title_emb_sample, *_ = sample_batch
    classifier_config = {
        "vector_size_title": title_emb_sample.shape[0],
        "vector_size_num": features_num_sample.shape[0],
        "scale": 3,
        "domain_vocab_size": len(vocabs['domain_vocab']),
        "tld_vocab_size": len(vocabs['tld_vocab']),
        "user_vocab_size": len(vocabs['user_vocab'])
    }
    classifier = ClassifierModel(**classifier_config).to(DEVICE)

    optimizer_class = optim.Adam(classifier.parameters(), lr=LEARNING_RATE_CLASS)
    pos_weight = torch.tensor([0.5 / 0.5]).to(DEVICE)
    criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_class_val_loss = float("inf")
    best_reg_val_loss = float("inf")

    for epoch in range(1, EPOCHS_CLASS+1):
        classifier.train()
        train_loss = 0

        for batch in tqdm(train_class_dataloader, desc=f"Epoch {epoch} Classifier [Train]"):

            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]
            optimizer_class.zero_grad()
            logits = classifier(features_num, title_emb, domain_idx, tld_idx, user_idx).squeeze()
            loss = criterion_class(logits, target)
            loss.backward()
            optimizer_class.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_class_dataloader)

        # ---- VALIDATION ----
        classifier.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_class_dataloader, desc=f"Epoch {epoch} Classifier [Val]"):
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]

                logits = classifier(features_num, title_emb, domain_idx, tld_idx, user_idx).squeeze()
                loss = criterion_class(logits, target)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == target).sum().item()
                total += target.size(0)

        avg_val_loss = val_loss / len(val_class_dataloader)

        if avg_val_loss < best_class_val_loss:
            best_class_val_loss = avg_val_loss
            model_path = os.path.join(model_dir, f"best_classifier_epoch_{epoch}.pth")
            torch.save({'model_state_dict': classifier.state_dict(), "config": classifier_config}, model_path)
            print(f"✓ Saved new best classifier at epoch {epoch} with val loss {avg_val_loss:.4f}")

        val_acc = correct / total

        writer.add_scalar("Classifier/Train_Loss", avg_train_loss, epoch)
        writer.add_scalar("Classifier/Val_Loss", avg_val_loss, epoch)
        writer.add_scalar("Classifier/Val_Acc", val_acc, epoch)

        print(f"✓ Classifier Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f}")

    ### ---------- REGRESSOR ----------
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    train_reg_dataset = PrecomputedNPZDataset(TRAIN_FILE, task="regression")
    val_reg_dataset = PrecomputedNPZDataset(VAL_FILE, task="regression")

    train_reg_dataloader = DataLoader(train_reg_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_reg_dataloader = DataLoader(val_reg_dataset, batch_size=BATCH_SIZE, shuffle=False)

    regressor_config = classifier_config
    regressor_config["num_quantiles"] = len(quantiles)
    regressor = QuantileRegressionModel(**regressor_config).to(DEVICE)

    optimizer_reg = optim.Adam(regressor.parameters(), lr=LEARNING_RATE_REG)
    criterion_reg = QuantileLoss(quantiles, device=DEVICE).to(DEVICE)
    # criterion_reg = nn.HuberLoss()

    for epoch in range(1, EPOCHS_REG+1):
        regressor.train()
        train_loss = 0

        for batch in tqdm(train_reg_dataloader, desc=f"Epoch {epoch} Regressor [Train]"):

            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]

            optimizer_reg.zero_grad()
            output = regressor(features_num, title_emb, domain_idx, tld_idx, user_idx)

            # loss = criterion_reg(output, target.unsqueeze(1))
            loss = criterion_reg(output, target)

            loss.backward()
            optimizer_reg.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_reg_dataloader)

        # ---- VALIDATION ----
        regressor.eval()
        val_loss = 0
        val_steps = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(val_reg_dataloader, desc=f"Epoch {epoch} Regressor [Val]"):
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]

                output = regressor(features_num, title_emb, domain_idx, tld_idx, user_idx)

                # loss = criterion_reg(output, target.unsqueeze(1))
                loss = criterion_reg(output, target)
                val_loss += loss.item()
                val_steps += 1

                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        avg_val_loss = val_loss / val_steps

        if avg_val_loss < best_reg_val_loss:
            best_reg_val_loss = avg_val_loss
            model_path = os.path.join(model_dir, f"best_quantile_regressor_epoch_{epoch}.pth")
            torch.save({'model_state_dict': regressor.state_dict(), "config": regressor_config}, model_path)
            print(f"✓ Saved new best regressor at epoch {epoch} with val loss {avg_val_loss:.4f}")

        # predictions = np.vstack(predictions)
        # targets = np.concatenate(targets)
        # predictions = np.vstack(predictions).squeeze()
        # targets = np.concatenate(targets)
        # mae = np.mean(np.abs(predictions - targets))

        predictions = np.vstack(predictions)  # shape (num_samples, num_quantiles)
        targets = np.concatenate(targets)     # shape (num_samples,)

        median_preds = predictions[:, 2]  # index 2 is 50th percentile for quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        mae = np.mean(np.abs(median_preds - targets))
        # mae = np.mean(np.abs(predictions - targets))

        print(f"P vs T {predictions[:5]} and {targets[:5]}")

        writer.add_scalar("QuantileRegressor/Train_Loss", avg_train_loss, epoch)
        writer.add_scalar("QuantileRegressor/Val_Loss", avg_val_loss, epoch)
        writer.add_scalar("QuantileRegressor/Val_MAE", mae, epoch)

        print(f"✓ Regressor Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, MAE {mae:.4f}")

    writer.close()
