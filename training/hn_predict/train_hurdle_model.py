import json
import logging
import os

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import utils
from models.hn_predict import ClassifierModel, QuantileRegressionModel, CLASSIFIER_MAX
from .dataloader import PrecomputedDataset
from .helpers import QuantileLoss

hyperparameters = {
    "batch_size": 8192,
    "epochs_classifier": 5,
    "epochs_regressor": 5,
    "lr_classifier": 1e-3,
    "lr_regressor": 1e-4,
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

    ### ---------- CLASSIFIER ----------
    train_class_dataloader = make_data_loader(train_dataset, shuffle=True)
    val_class_dataloader = make_data_loader(val_dataset, shuffle=False)
    logging.info("Created data loaders")

    sample_batch = train_dataset[0]
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
    logging.info("Built model")

    # after: train_class_dataset, train_class_dataloader = make_data_loader(...)
    # hurdle is CLASSIFIER_MAX (e.g., 6 for ≥7)
    with torch.no_grad():
        y = train_dataset.targets  # torch tensor already in RAM
        pos = (y > CLASSIFIER_MAX).sum().item()
        tot = y.numel()
        neg = tot - pos
        pos_w = (neg / max(1, pos)) if pos > 0 else 1.0

    pos_weight = torch.tensor([pos_w], device=DEVICE, dtype=torch.float32)
    criterion_class = BCEWithLogitsLoss(pos_weight=pos_weight)
    logging.info(f"pos_weight (neg/pos) at hurdle {CLASSIFIER_MAX}: {pos_w:.3f}")

    run = wandb.init(
        entity="tyronenicholas",
        project="hn_predict",
        config=hyperparameters,
    )

    best_class_val_loss = float("inf")
    best_reg_val_loss = float("inf")

    for epoch in range(1, hyperparameters["epochs_classifier"] + 1):
        classifier.train()
        train_loss = 0

        for batch in tqdm(
            train_class_dataloader, desc=f"Epoch {epoch} Classifier [Train]"
        ):

            features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                b.to(DEVICE, non_blocking=True) for b in batch
            ]
            target_cls = (target > CLASSIFIER_MAX).float()
            optimizer_class.zero_grad()
            logits = classifier(
                features_num, title_emb, domain_idx, tld_idx, user_idx
            ).squeeze(1)
            loss = criterion_class(logits, target_cls)
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
                target_cls = (target > CLASSIFIER_MAX).float()
                logits = classifier(
                    features_num, title_emb, domain_idx, tld_idx, user_idx
                ).squeeze(1)
                loss = criterion_class(logits, target_cls)
                val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(target_cls.cpu())

        all_logits = torch.cat(all_logits).view(-1)
        all_targets = torch.cat(all_targets).view(-1)

        probs = torch.sigmoid(all_logits)
        pos_rate = all_targets.mean().item()

        ths = torch.linspace(0.05, 0.95, steps=19)
        best = dict(th=0.5, f1=0.0, acc=0.0, p=0.0, r=0.0, pred_rate=0.0)
        for th in ths:
            preds = (probs > th).float()
            tp = ((preds == 1) & (all_targets == 1)).sum().item()
            tn = ((preds == 0) & (all_targets == 0)).sum().item()
            fp = ((preds == 1) & (all_targets == 0)).sum().item()
            fn = ((preds == 0) & (all_targets == 1)).sum().item()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = (
                0.0
                if (precision + recall) == 0
                else 2 * precision * recall / (precision + recall)
            )
            acc = (tp + tn) / max(1, tp + tn + fp + fn)
            pred_rate = preds.mean().item()
            if f1 > best["f1"]:
                best = dict(
                    th=float(th),
                    f1=f1,
                    acc=acc,
                    p=precision,
                    r=recall,
                    pred_rate=pred_rate,
                )

        avg_val_loss = val_loss / len(val_class_dataloader)
        final_preds = (probs > best["th"]).float()
        val_acc = (final_preds == all_targets).float().mean().item()

        run.log(
            {
                "Classifier/Val_Loss": avg_val_loss,
                "Classifier/Val_F1": best["f1"],
                "Classifier/Val_Acc": val_acc,
                "Classifier/Val_Thresh": best["th"],
                "Classifier/Val_PosRate": pos_rate,
                "Classifier/Val_PredRate": best["pred_rate"],
            }
        )
        logging.info(
            f"✓ Classifier Epoch {epoch}: Train Loss {avg_train_loss:.4f}, "
            f"Val Loss {avg_val_loss:.4f}, Acc {val_acc:.4f}, MacroF1 {best['f1']:.3f}"
        )
        vals, counts = torch.unique(all_targets, return_counts=True)
        logging.info(
            f"Val class histogram: "
            + ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
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
    pos_idx_train = (train_dataset.targets > CLASSIFIER_MAX).nonzero(as_tuple=True)[0]
    pos_idx_val = (val_dataset.targets > CLASSIFIER_MAX).nonzero(as_tuple=True)[0]
    train_reg_dataloader = make_data_loader(
        Subset(train_dataset, pos_idx_train.tolist()), shuffle=True
    )
    val_reg_dataloader = make_data_loader(
        Subset(val_dataset, pos_idx_val.tolist()), shuffle=False
    )

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
            target_excess = target - CLASSIFIER_MAX - 1.0
            target_log = torch.log1p(target_excess)
            output = regressor(features_num, title_emb, domain_idx, tld_idx, user_idx)
            loss = criterion_reg(output, target_log)

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

                target_excess = target - CLASSIFIER_MAX - 1.0
                target_log = torch.log1p(target_excess)
                output = regressor(
                    features_num, title_emb, domain_idx, tld_idx, user_idx
                )
                loss = criterion_reg(output, target_log)
                val_loss += loss.item()
                val_steps += 1

                predictions.append(output.cpu().numpy())
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
