import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import utils
from models.hn_predict import ClassifierModel, QuantileRegressionModel, CLASSIFIER_MAX
from .dataloader import PrecomputedNPZDataset
from .helpers import QuantileLoss

hyperparameters = {
    "batch_size": 8192,
    "epochs_classifier": 5,
    "epochs_regressor": 5,
    "lr_classifier": 1e-3,
    "lr_regressor": 1e-4,
}
DEVICE = utils.get_device()

TRAIN_FILE = "data/train.npz"
VAL_FILE = "data/val.npz"
VOCAB_FILE = "data/train_vocab.json"

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def make_data_loader(npz_path, task, shuffle, sampler=None):
    dataset = PrecomputedNPZDataset(npz_path, task=task)
    return dataset, DataLoader(
        dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
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

    ### ---------- CLASSIFIER ----------
    train_class_dataset = PrecomputedNPZDataset(TRAIN_FILE, task="classification")

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

    # Compute class weights (inverse frequency) once from train set to soften imbalance
    with torch.no_grad():
        y_train = train_class_dataset.targets
        y_classes = torch.clamp(
            y_train.long(), 0, CLASSIFIER_MAX + 1
        )  # 0..6 exact, 7 for >=7
        hist = torch.bincount(y_classes, minlength=(CLASSIFIER_MAX + 2)).to(
            torch.float32
        )
    logging.info(f"Class counts: {hist.tolist()}")
    inv = 1.0 / torch.clamp(hist, 1.0)
    inv = torch.clamp(inv, 1.0, 50.0)  # cap extremes
    sample_weights = inv[y_classes].to(torch.float64)
    sampler = WeightedRandomSampler(
        weights=sample_weights.detach().to(torch.double),
        num_samples=sample_weights.numel(),
        replacement=True,
    )

    beta = 0.999
    beta_t = torch.tensor(beta, dtype=hist.dtype, device=hist.device)

    # Effective number of samples per class (Cui et al.)
    effective_num = (1.0 - torch.pow(beta_t, hist)) / (1.0 - beta_t)

    # Raw weights = 1 / effective_num, then normalize and clip
    raw_class_weights = 1.0 / torch.clamp(effective_num, min=1e-6)
    normalized_class_weights = raw_class_weights / raw_class_weights.mean()
    clipped_class_weights = torch.clamp(normalized_class_weights, min=0.5, max=5.0)

    class_weights = clipped_class_weights.to(device=DEVICE, dtype=torch.float32)
    logging.info(f"Class weights: {[round(x, 3) for x in class_weights.tolist()]}")

    criterion_class = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    run = wandb.init(
        entity="tyronenicholas",
        project="hn_predict",
        config=hyperparameters,
    )

    train_class_dataset, train_class_dataloader = make_data_loader(
        TRAIN_FILE, task="classification", shuffle=False, sampler=sampler
    )
    _, val_class_dataloader = make_data_loader(
        VAL_FILE, task="classification", shuffle=False
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
            target_cls = torch.clamp(target.long(), max=CLASSIFIER_MAX + 1)
            optimizer_class.zero_grad()
            logits = classifier(features_num, title_emb, domain_idx, tld_idx, user_idx)
            loss = criterion_class(logits, target_cls)
            loss.backward()
            optimizer_class.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_class_dataloader)

        # ---- VALIDATION ----
        classifier.eval()
        val_loss = 0
        correct, total = 0, 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(
                val_class_dataloader, desc=f"Epoch {epoch} Classifier [Val]"
            ):
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [
                    b.to(DEVICE, non_blocking=True) for b in batch
                ]
                target_cls = torch.clamp(target.long(), max=CLASSIFIER_MAX + 1)
                logits = classifier(
                    features_num, title_emb, domain_idx, tld_idx, user_idx
                )
                loss = criterion_class(logits, target_cls)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(target_cls.cpu())

        all_preds = torch.cat(all_preds).view(-1).cpu()
        all_targets = torch.cat(all_targets).view(-1).cpu()

        confusion_matrix = torch.zeros(
            CLASSIFIER_MAX + 2, CLASSIFIER_MAX + 2, dtype=torch.int64
        )
        for tgt, pred in zip(all_targets, all_preds):
            confusion_matrix[tgt, pred] += 1
        logging.info(
            "Confusion (rows=true, cols=pred):\n"
            + "\n".join(
                " ".join(f"{int(x):7d}" for x in row.tolist())
                for row in confusion_matrix
            )
        )

        f1s = []
        for cls in range(CLASSIFIER_MAX + 2):
            true_pos = confusion_matrix[cls, cls].item()
            false_pos = (confusion_matrix[:, cls].sum() - true_pos).item()
            false_neg = (confusion_matrix[cls, :].sum() - true_pos).item()
            prec = true_pos / max(1, true_pos + false_pos)
            rec = true_pos / max(1, true_pos + false_neg)
            f1 = 2 * prec * rec / max(1e-8, prec + rec)
            f1s.append(f1)
            logging.info(f"class {cls}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

        avg_val_loss = val_loss / len(val_class_dataloader)
        acc = (all_preds == all_targets).float().mean().item()
        macro_f1 = float(np.mean(f1s))

        run.log(
            {
                "Classifier/Val_Loss": avg_val_loss,
                "Classifier/Val_Acc": acc,
                "Classifier/Val_MacroF1": macro_f1,
            }
        )
        logging.info(
            f"✓ Classifier Epoch {epoch}: Train Loss {avg_train_loss:.4f}, "
            f"Val Loss {avg_val_loss:.4f}, Acc {acc:.4f}, MacroF1 {macro_f1:.3f}"
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
            mask = target > CLASSIFIER_MAX
            if not mask.any():
                continue
            target_excess = target[mask] - CLASSIFIER_MAX - 1.0
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

                mask = target > CLASSIFIER_MAX
                if not mask.any():
                    continue
                target_excess = target[mask] - CLASSIFIER_MAX - 1.0
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
