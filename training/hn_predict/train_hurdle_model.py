import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from common import utils
from models.hn_predict import QuantileRegressionModel
from models.hn_predict_utils import TRAINING_VOCAB_PATH, VOCAB_PATH
from .dataloader import PrecomputedDataset
from .helpers import QuantileLoss

hyperparameters = {
    "scale": 4,
    "batch_size": 8192,
    "epochs": 5,
    "learning_rate": 1e-3,
    "text_learning_rate": 3e-3,
}
DEVICE = utils.get_device()

TRAIN_FILE = "data/train.pt"
VAL_FILE = "data/val.pt"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def collate_fn(batch):
    # Separate the different parts of the data
    (
        features_num,
        title_indices,
        domain_indices,
        tld_indices,
        user_indices,
        targets,
    ) = zip(*batch)

    # Stack the fixed-size data
    features_num = torch.stack(features_num)
    domain_indices = torch.stack(domain_indices)
    tld_indices = torch.stack(tld_indices)
    user_indices = torch.stack(user_indices)
    targets = torch.stack(targets)

    # --- Pad the variable-length title sequences ---
    title_tensors = [
        torch.tensor(t if len(t) > 0 else [0], dtype=torch.long) for t in title_indices
    ]

    title_indices_padded = pad_sequence(
        title_tensors, batch_first=True, padding_value=0  # 0 = PAD
    )

    return (
        features_num,
        title_indices_padded,
        domain_indices,
        tld_indices,
        user_indices,
        targets,
    )


def make_data_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=shuffle,
        pin_memory=True,
        pin_memory_device="cuda",
        num_workers=8,
        prefetch_factor=4,
        collate_fn=collate_fn,
    )


model_dir = "data"


def main():
    utils.setup_logging()
    logging.info("Starting run")
    os.makedirs(model_dir, exist_ok=True)

    with open(TRAINING_VOCAB_PATH, "r") as f:
        vocabs = json.load(f)

    with open(VOCAB_PATH, "r") as f:
        word_vocab = json.load(f)

    train_dataset = PrecomputedDataset(TRAIN_FILE)
    val_dataset = PrecomputedDataset(VAL_FILE)
    logging.info("Created data sets")

    all_targets = []
    for i in range(len(train_dataset)):
        _, _, _, _, _, target = train_dataset[i]
        all_targets.append(target.item())

    all_targets = np.array(all_targets)

    # Compute custom quantiles based on actual distribution
    # For example: 46%, 66%, 75%, 90%, 97%
    quantile_probs = [0.46, 0.66, 0.75, 0.90, 0.97]
    quantiles = np.quantile(all_targets, quantile_probs)

    logging.info(f"Custom quantile cut points: {quantiles}")
    sample_batch = train_dataset[0]
    features_num_sample, *_ = sample_batch
    config = {
        "vector_size_num": features_num_sample.shape[0],
        "scale": hyperparameters["scale"],
        "domain_vocab_size": len(vocabs["domain_vocab"]),
        "tld_vocab_size": len(vocabs["tld_vocab"]),
        "user_vocab_size": len(vocabs["user_vocab"]),
        "word_vocab_size": len(word_vocab),
        "num_quantiles": len(quantiles),
    }
    run = wandb.init(
        entity="tyronenicholas",
        project="hn_predict",
        config=hyperparameters,
    )

    best_val_loss = float("inf")

    regressor = QuantileRegressionModel(**config).to(DEVICE)

    text_params = list(regressor.word_embedding.parameters())
    other_params = [
        p for n, p in regressor.named_parameters() if not n.startswith("word_embedding")
    ]

    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": hyperparameters["learning_rate"]},
            {"params": text_params, "lr": hyperparameters["text_learning_rate"]},
        ],
        weight_decay=1e-2,
    )

    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer, 1, hyperparameters["epochs"], min_lr_rate=0.25
    )

    criterion = QuantileLoss(quantile_probs, device=DEVICE).to(DEVICE)

    train_dataloader = make_data_loader(train_dataset, shuffle=True)
    val_dataloader = make_data_loader(val_dataset, shuffle=False)

    for epoch in range(1, hyperparameters["epochs"] + 1):
        regressor.train()
        train_loss = 0
        train_steps = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]"):

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
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]"):
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
            model_path = os.path.join(model_dir, f"best_quantile.pth")
            torch.save(
                {
                    "model_state_dict": regressor.state_dict(),
                    "config": config,
                    "quantile_probs": quantile_probs,
                },
                model_path,
            )
            logging.info(
                f"✓ Saved new best regressor at epoch {epoch} with val loss {avg_val_loss:.4f}"
            )

        predictions = np.vstack(predictions)  # shape (num_samples, num_quantiles)
        predictions = np.maximum.accumulate(predictions, axis=1)
        # predictions: (N, Q) in log space
        targets_lin = np.concatenate(targets)  # shape (num_samples,)
        probs = np.array(
            quantile_probs, dtype=np.float32
        )  # e.g., [0.46, 0.66, 0.75, 0.90, 0.97]
        preds_log = predictions

        # find neighbors around 0.50
        if (probs == 0.5).any():
            median_log_preds = preds_log[:, np.where(probs == 0.5)[0][0]]
        else:
            # idx of last prob <= 0.5 and first prob >= 0.5
            lo = probs[probs <= 0.5].argmax()  # index of 0.46
            hi = probs[probs >= 0.5].argmin()  # index of 0.66
            p_lo, p_hi = probs[lo], probs[hi]
            y_lo, y_hi = preds_log[:, lo], preds_log[:, hi]
            # linear interp in prob space (still in log-target space)
            t = (0.5 - p_lo) / max(1e-6, (p_hi - p_lo))
            median_log_preds = y_lo + t * (y_hi - y_lo)

        targets_log = np.log1p(targets_lin)
        mae_log = np.mean(np.abs(median_log_preds - targets_log))
        median_lin_preds = np.expm1(median_log_preds)
        mae_lin = np.mean(np.abs(median_lin_preds - targets_lin))

        last_lr = scheduler.get_last_lr()[0]
        score_buckets = {
            "B1 (<=1)": (-1, 1, 0.37),  # 0.46
            "B2 (2–3)": (2, 3, 0.27),  # 0.30
            "B3 (4–6)": (4, 6, 0.08),  # 0.09
            "B4 (7–16)": (7, 16, 0.08),  # 0.06
            "B5 (17–99)": (17, 99, 0.09),  # 0.07
            "B6 (≥100)": (100, np.inf, 0.11),  # 0.08
        }

        bucket_maes = {}
        weighted_mae = 0
        for name, (low, high, weight) in score_buckets.items():
            mask = (targets_lin >= low) & (targets_lin <= high)
            if mask.sum() > 0:
                bucket_mae = np.mean(np.abs(median_lin_preds[mask] - targets_lin[mask]))
                bucket_maes[f"Val_MAE/{name}"] = bucket_mae
                logging.info(
                    f"  ... MAE for {name}: {bucket_mae:.4f} (on {mask.sum()} samples)"
                )
                weighted_mae += weight * bucket_mae

        qs = np.array(quantile_probs, dtype=np.float32)
        covered = (targets_log[:, None] <= predictions).mean(
            axis=0
        )  # fraction ≤ each quantile
        coverage = {}
        for q, c in zip(qs, covered):
            coverage[f"Coverage/{q:.2f}"] = float(c)
            logging.info(f"Coverage/{q:.2f} {float(c):.4f}")

        run.log(
            {
                "QuantileRegressor/Train_Loss": avg_train_loss,
                "QuantileRegressor/Val_Loss": avg_val_loss,
                "QuantileRegressor/Val_MAE_Log": mae_log,
                "QuantileRegressor/Val_MAE_Lin": mae_lin,
                "QuantileRegressor/Val_MAE_Weighted": weighted_mae,
                "QuantileRegressor/LearningRate": last_lr,
                **bucket_maes,
                **coverage,
            }
        )

        logging.info(
            f"✓ Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, MAE Log {mae_log:.4f} MAE Lin {mae_lin:.4f} MAE Weighted {weighted_mae:.4f} LearningRate: {last_lr:.4f}"
        )

    run.finish(0)
    plot_predictions(median_lin_preds, targets_lin)


def plot_predictions(median_lin_preds, targets_lin):
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
    max_val = max(targets_to_plot.max().item(), preds_to_plot.max().item())
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


if __name__ == "__main__":
    main()
