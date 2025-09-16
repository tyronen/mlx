import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import utils
from models.hn_predict import ClassifierModel, QuantileRegressionModel
from .dataloader import PrecomputedNPZDataset
from .helpers import QuantileLoss

hyperparameters = {
    "batch_size": 8192,
    "epochs_classifier": 5,
    "epochs_regressor": 5,
    "lr_classifier": 1e-3,
    "lr_regressor": 1e-4,
    "threshold": 6,
}
DEVICE = utils.get_device()

TRAIN_FILE = "data/train.npz"
VAL_FILE = "data/val.npz"
VOCAB_FILE = "data/train_vocab.json"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def estimate_pos_weight(dset):
    y = dset.targets  # tensor on CPU
    pos = (y > hyperparameters["threshold"]).sum().item()
    tot = y.numel()
    neg = max(0, tot - pos)
    w = (neg / pos) if pos > 0 else 1.0
    # clamp to something sane
    return float(np.clip(w, 0.5, 50.0))


def make_data_loader(npz_path, task, shuffle):
    dataset = PrecomputedNPZDataset(npz_path, task=task)
    return dataset, DataLoader(
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
    model_dir = "data"
    os.makedirs(model_dir, exist_ok=True)
    run = wandb.init(
        entity="tyronenicholas",
        project="hn_predict",
        config=hyperparameters,
    )

    with open(VOCAB_FILE, "r") as f:
        vocabs = json.load(f)

    ### ---------- CLASSIFIER ----------
    train_class_dataset, train_class_dataloader = make_data_loader(
        TRAIN_FILE, task="classification", shuffle=True
    )
    _, val_class_dataloader = make_data_loader(
        VAL_FILE, task="classification", shuffle=False
    )

    sample_batch = train_class_dataset[0]
    features_num_sample, title_emb_sample, *_ = sample_batch
    classifier_config = {
        "vector_size_title": title_emb_sample.shape[0],
        "vector_size_num": features_num_sample.shape[0],
        "scale": 3,
        "domain_vocab_size": len(vocabs["domain_vocab"]),
        "tld_vocab_size": len(vocabs["tld_vocab"]),
        "user_vocab_size": len(vocabs["user_vocab"]),
    }
    classifier = ClassifierModel(**classifier_config).to(DEVICE)

    optimizer_class = optim.Adam(
        classifier.parameters(), lr=hyperparameters["lr_classifier"]
    )
    w = estimate_pos_weight(train_class_dataset)
    pos_weight = torch.tensor([w], device=DEVICE)
    criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logging.info(f"Using pos_weight={w:.3f} (neg/pos)")

    best_class_val_loss = float("inf")
    best_reg_val_loss = float("inf")

    for epoch in range(1, hyperparameters["epochs_classifer"] + 1):
        classifier.train()
        train_loss = 0

        for batch in tqdm(
            train_class_dataloader, desc=f"Epoch {epoch} Classifier [Train]"
        ):

            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                b.to(DEVICE, non_blocking=True) for b in batch
            ]
            target = (target > hyperparameters["threshold"]).float()
            optimizer_class.zero_grad()
            logits = classifier(
                features_num, title_emb, domain_idx, tld_idx, user_idx
            ).squeeze()
            loss = criterion_class(logits, target)
            loss.backward()
            optimizer_class.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_class_dataloader)

        # ---- VALIDATION ----
        classifier.eval()
        val_loss = 0
        correct, total = 0, 0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(
                val_class_dataloader, desc=f"Epoch {epoch} Classifier [Val]"
            ):
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                    b.to(DEVICE, non_blocking=True) for b in batch
                ]
                target = (target > hyperparameters["threshold"]).float()
                logits = classifier(
                    features_num, title_emb, domain_idx, tld_idx, user_idx
                ).squeeze()
                loss = criterion_class(logits, target)
                val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(target.cpu())

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        probs = torch.sigmoid(all_logits)
        ths = torch.linspace(0.05, 0.95, steps=19)
        best_f1, best_th = 0.0, 0.5
        for th in ths:
            preds = (probs > th).float()
            tp = ((preds == 1) & (all_targets == 1)).sum().item()
            fp = ((preds == 1) & (all_targets == 0)).sum().item()
            fn = ((preds == 0) & (all_targets == 1)).sum().item()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)

        # compute metrics at best threshold for logging
        final_preds = (probs > best_th).float()
        val_acc = (final_preds == all_targets).float().mean().item()

        avg_val_loss = val_loss / len(val_class_dataloader)
        run.log(
            {
                "Classifier/Val_F1": best_f1,
                "Classifier/Val_Thresh": best_th,
                "Classifier/Val_Loss": avg_val_loss,
                "Classifier/Val_Acc": val_acc,
            }
        )

        logging.info(
            f"✓ Classifier Epoch {epoch}: Train Loss {avg_train_loss:.4f}, "
            f"Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f} (best_th={best_th:.2f}, F1={best_f1:.3f})"
        )

        if avg_val_loss < best_class_val_loss:
            best_class_val_loss = avg_val_loss
            model_path = os.path.join(model_dir, f"best_classifier_epoch_{epoch}.pth")
            torch.save(
                {
                    "model_state_dict": classifier.state_dict(),
                    "config": classifier_config,
                },
                model_path,
            )
            logging.info(
                f"✓ Saved new best classifier at epoch {epoch} with val loss {avg_val_loss:.4f}"
            )

    ### ---------- REGRESSOR ----------
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    _, train_reg_dataloader = make_data_loader(
        TRAIN_FILE, task="regression", shuffle=True
    )
    _, val_reg_dataloader = make_data_loader(VAL_FILE, task="regression", shuffle=False)

    regressor_config = classifier_config
    regressor_config["num_quantiles"] = len(quantiles)
    regressor = QuantileRegressionModel(**regressor_config).to(DEVICE)

    optimizer_reg = optim.Adam(
        regressor.parameters(), lr=hyperparameters["lr_regressor"]
    )
    criterion_reg = QuantileLoss(quantiles, device=DEVICE).to(DEVICE)

    for epoch in range(1, hyperparameters["epochs_regressor"] + 1):
        regressor.train()
        train_loss = 0
        train_steps = 0
        for batch in tqdm(
            train_reg_dataloader, desc=f"Epoch {epoch} Regressor [Train]"
        ):

            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                b.to(DEVICE, non_blocking=True) for b in batch
            ]

            optimizer_reg.zero_grad()
            mask = target > hyperparameters["threshold"]
            if not mask.any():
                continue
            target_excess = target[mask] - hyperparameters["threshold"]
            target_log = torch.log1p(target_excess)
            output = regressor(features_num, title_emb, domain_idx, tld_idx, user_idx)
            output_pos = output[mask]
            loss = criterion_reg(output_pos, target_log)

            loss.backward()
            optimizer_reg.step()
            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(1, train_steps)

        # ---- VALIDATION ----
        regressor.eval()
        val_loss = 0
        val_steps = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(
                val_reg_dataloader, desc=f"Epoch {epoch} Regressor [Val]"
            ):
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                    b.to(DEVICE, non_blocking=True) for b in batch
                ]

                mask = target > hyperparameters["threshold"]
                if not mask.any():
                    continue
                target_excess = target[mask] - hyperparameters["threshold"]
                target_log = torch.log1p(target_excess)
                output = regressor(
                    features_num, title_emb, domain_idx, tld_idx, user_idx
                )
                output_pos = output[mask]
                loss = criterion_reg(output_pos, target_log)
                val_loss += loss.item()
                val_steps += 1

                predictions.append(output_pos.cpu().numpy())
                targets.append(target_excess.cpu().numpy())

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

        if avg_val_loss < best_reg_val_loss:
            best_reg_val_loss = avg_val_loss
            model_path = os.path.join(
                model_dir, f"best_quantile_regressor_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "model_state_dict": regressor.state_dict(),
                    "config": regressor_config,
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

        run.log(
            {
                "QuantileRegressor/Train_Loss": avg_train_loss,
                "QuantileRegressor/Val_Loss": avg_val_loss,
                "QuantileRegressor/Val_MAE_Log": mae_log,
                "QuantileRegressor/Val_MAE_Lin": mae_lin,
            }
        )

        logging.info(
            f"✓ Regressor Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, MAE Log {mae_log:.4f} MAE Lin {mae_lin:.4f}"
        )

    run.finish(0)
