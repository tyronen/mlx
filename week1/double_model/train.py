import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, load_embeddings, get_device, QuantileLoss
from dataloader import PrecomputedNPZDataset
from model import FullModel
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from torchinfo import summary

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

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

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Load vocab sizes from vocab file
    with open(VOCAB_FILE, 'r') as f:
        vocabs = json.load(f)

    domain_vocab_size = len(vocabs['domain_vocab'])
    tld_vocab_size = len(vocabs['tld_vocab'])
    user_vocab_size = len(vocabs['user_vocab'])

    # Dataset
    train_dataset = PrecomputedNPZDataset(TRAIN_FILE, time_decay=None)
    val_dataset = PrecomputedNPZDataset(VAL_FILE, time_decay=None)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    sample_batch = train_dataset[0]
    features_num_sample, title_emb_sample, *_ = sample_batch
    vector_size_title = title_emb_sample.shape[0]
    vector_size_num = features_num_sample.shape[0]

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    model = FullModel(vector_size_num=vector_size_num,
                    vector_size_title=vector_size_title,
                    scale=3,
                    domain_vocab_size=domain_vocab_size,
                    tld_vocab_size=tld_vocab_size,
                    user_vocab_size=user_vocab_size,
                    num_quantiles=len(quantiles)).to(DEVICE)

    # Generate a summary
    summary(model, input_data=(
        torch.randn(1, vector_size_num).to(DEVICE),
        torch.randn(1, vector_size_title).to(DEVICE),
        torch.randint(0, domain_vocab_size, (1,)).to(DEVICE),
        torch.randint(0, tld_vocab_size, (1,)).to(DEVICE),
        torch.randint(0, user_vocab_size, (1,)).to(DEVICE),
    ))

    # Loss and optimizer
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(reduction="mean", delta=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Define quantiles and loss function
    criterion = QuantileLoss(quantiles,device=DEVICE).to(DEVICE)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_dataloader, total=len(train_dataloader),
                             desc=f"Epoch {epoch} [Train]", unit="batch", dynamic_ncols=True)

        for batch in progress_bar:
            
            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            output = model(features_num, title_emb, domain_idx, tld_idx, user_idx)
            # loss = criterion(output, target.unsqueeze(1)).mean()
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            progress_bar.update()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / train_steps

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        predictions = []
        targets = []

        val_progress = tqdm(val_dataloader, total=len(val_dataloader),
                             desc=f"Epoch {epoch} [Val]", unit="batch", dynamic_ncols=True)

        with torch.no_grad():
            for batch in val_progress:
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]

                output = model(features_num, title_emb, domain_idx, tld_idx, user_idx)
                loss = criterion(output, target)

                val_loss += loss.item()
                val_steps += 1

                # Append full output (do NOT flatten!)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        # Stack predictions and targets after loop
        predictions = np.vstack(predictions)   # shape: (total_samples, num_quantiles)
        targets = np.concatenate(targets)      # shape: (total_samples,)

        # Extract 50th percentile (index 1)
        median_preds = predictions[:, 1]

        print(f"predictions {median_preds[:5]} and target {targets[:5]}")

        # Compute MAE on median predictions
        mae = np.mean(np.abs(median_preds - targets))

        # # R² on log scale
        # ss_res_log = np.sum((targets - predictions) ** 2)
        # ss_tot_log = np.sum((targets - np.mean(targets)) ** 2)
        # r2_log = 1 - ss_res_log / ss_tot_log

        # # R² on real scale (inverse transform)
        # predictions_real = 10**predictions - 1
        # targets_real = 10**targets - 1

        # print(predictions_real[:5], targets_real[:5])

        # ss_res_real = np.sum((targets_real - predictions_real) ** 2)
        # ss_tot_real = np.sum((targets_real - np.mean(targets_real)) ** 2)
        # r2_real = 1 - ss_res_real / ss_tot_real

        # Write to TensorBoard
        # writer.add_scalar("Metrics/R2_log", r2_log, epoch)
        # writer.add_scalar("Metrics/R2_real", r2_real, epoch)
        avg_val_loss = val_loss / val_steps

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/MAE", mae, epoch)

        print(f"✓ Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MAE: {mae:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(model_dir, f'best_model_{epoch}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
            }, model_path)
            print(f"✓ Saved new best model at epoch {epoch} with val loss {avg_val_loss:.4f}")

    writer.close()
