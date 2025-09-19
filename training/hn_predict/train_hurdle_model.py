import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from common import utils
from models.hn_predict import QuantileRegressionModel
from .dataloader import PrecomputedDataset
from .helpers import QuantileLoss

hyperparameters = {
    "batch_size": 8192,
    "epochs_classifier": 5,
    "epochs_regressor": 5,
    "lr_classifier": 1e-2,
    "lr_regressor": 1e-3,
}
DEVICE = utils.get_device()

TRAIN_FILE = "data/train.pt"
VAL_FILE = "data/val.pt"
VOCAB_FILE = "data/train_vocab.json"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def make_data_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=shuffle,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=8,
        prefetch_factor=4,
    )


if __name__ == "__main__":
    utils.setup_logging()
    logging.info("Starting run")
    model_dir = "data"
    os.makedirs(model_dir, exist_ok=True)

    with open(VOCAB_FILE, "r") as f:
        vocabs = json.load(f)

    train_dataset = PrecomputedDataset(TRAIN_FILE)
    val_dataset = PrecomputedDataset(VAL_FILE)
    logging.info("Created data loaders")

    sample_batch = train_dataset[0]
    features_num_sample, title_emb_sample, *_ = sample_batch
    config = {
        "vector_size_title": title_emb_sample.shape[0],
        "vector_size_num": features_num_sample.shape[0],
        "scale": 3,
        "domain_vocab_size": len(vocabs["domain_vocab"]),
        "tld_vocab_size": len(vocabs["tld_vocab"]),
        "user_vocab_size": len(vocabs["user_vocab"]),
    }
    run = wandb.init(
        entity="tyronenicholas",
        project="hn_predict",
        config=hyperparameters,
    )

    best_val_loss = float("inf")

    ### ---------- REGRESSOR ----------
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    train_dataloader = make_data_loader(train_dataset, shuffle=True)
    val_dataloader = make_data_loader(val_dataset, shuffle=False)

    config["num_quantiles"] = len(quantiles)
    regressor = QuantileRegressionModel(**config).to(DEVICE)

    optimizer = optim.Adam(regressor.parameters(), lr=hyperparameters["lr_regressor"])
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, 1, hyperparameters["epochs_regressor"], min_lr_rate=0.25
    )

    criterion = QuantileLoss(quantiles, device=DEVICE).to(DEVICE)

    for epoch in range(1, hyperparameters["epochs_regressor"] + 1):
        regressor.train()
        train_loss = 0
        train_steps = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} Regressor [Train]"):

            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                b.to(DEVICE, non_blocking=True) for b in batch
            ]

            optimizer.zero_grad()
            target_log = torch.log1p(target)
            output = regressor(features_num, title_emb, domain_idx, tld_idx, user_idx)
            loss = criterion(output, target_log)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, train_steps)

        # ---- VALIDATION ----
        regressor.eval()
        val_loss = 0
        val_steps = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} Regressor [Val]"):
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                    b.to(DEVICE, non_blocking=True) for b in batch
                ]

                target_log = torch.log1p(target)
                output = regressor(
                    features_num, title_emb, domain_idx, tld_idx, user_idx
                )
                loss = criterion(output, target_log)
                val_loss += loss.item()
                val_steps += 1

                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(
                model_dir, f"best_quantile_regressor_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "model_state_dict": regressor.state_dict(),
                    "config": config,
                },
                model_path,
            )
            logging.info(
                f"✓ Saved new best regressor at epoch {epoch} with val loss {avg_val_loss:.4f}"
            )

        predictions = np.vstack(predictions)  # shape (num_samples, num_quantiles)
        targets_lin = np.concatenate(targets)  # shape (num_samples,)
        median_log_preds = predictions[:, 2]  # 50th pct in log space
        targets_log = np.log1p(targets_lin)
        mae_log = np.mean(np.abs(median_log_preds - targets_log))
        median_lin_preds = np.expm1(median_log_preds)  # back to linear
        mae_lin = np.mean(np.abs(median_lin_preds - targets_lin))
        last_lr = scheduler.get_last_lr()[0]
        run.log(
            {
                "QuantileRegressor/Train_Loss": avg_train_loss,
                "QuantileRegressor/Val_Loss": avg_val_loss,
                "QuantileRegressor/Val_MAE_Log": mae_log,
                "QuantileRegressor/Val_MAE_Lin": mae_lin,
                "QuantileRegressor/LearningRate": last_lr,
            }
        )
        # after computing predictions/targets_log
        qs = np.array(quantiles, dtype=np.float32)
        covered = (targets_log[:, None] <= predictions).mean(
            axis=0
        )  # fraction ≤ each quantile
        for q, c in zip(qs, covered):
            run.log({f"QuantileRegressor/Coverage@{q:.2f}": float(c)})

        logging.info(
            f"✓ Regressor Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, MAE Log {mae_log:.4f} MAE Lin {mae_lin:.4f} LearningRate: {last_lr:.4f}"
        )

    run.finish(0)

    ### ---------- PLOTTING ----------
    logging.info("Generating prediction vs. actual plot...")

    # Use the predictions and targets from the final validation epoch
    # Convert numpy arrays to torch tensors for easier plotting
    preds_to_plot = torch.from_numpy(median_lin_preds)
    targets_to_plot = torch.from_numpy(targets_lin)

    plt.figure(figsize=(10, 10))

    # Plotting on a log-log scale is best for skewed distributions like this
    plt.scatter(
        targets_to_plot + 1, preds_to_plot + 1, alpha=0.1
    )  # Use alpha for overplotting

    # Add a line for reference (perfect prediction)
    max_val = max(targets_to_plot.max(), preds_to_plot.max())
    plt.plot(
        [1, max_val + 1],
        [1, max_val + 1],
        color="red",
        linestyle="--",
        label="Perfect Prediction",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual Score + 1 (Log Scale)")
    plt.ylabel("Predicted Score + 1 (Log Scale)")
    plt.title("Predicted vs. Actual Scores (Final Epoch)")
    plt.legend()
    plt.grid(True, which="both", ls="--", c="0.7")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(model_dir, "predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=300)
    logging.info(f"✓ Plot saved to {plot_path}")
