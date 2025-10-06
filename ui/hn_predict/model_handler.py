import json
import logging
import os

import numpy as np
import torch

from common import utils
from models import hn_predict_utils
from models.hn_predict import QuantileRegressionModel
from models.hn_predict_utils import (
    SCALER_PATH,
    load_user_data,
)

REGRESSOR_PATH = os.getenv("REGRESSION_MODEL_PATH", "data/best_quantile.pth")

utils.setup_logging()


def get_full_model_preprocessor():
    """Load necessary data"""
    # Use the same vocab mapping as used during training
    with open(hn_predict_utils.WORD_TO_IX_PATH, "r") as f:
        w2i = json.load(f)
    if "UNK" not in w2i:
        w2i["UNK"] = 0

    # Load categorical vocabs
    with open(hn_predict_utils.TRAINING_VOCAB_PATH, "r") as f:
        vocabs = json.load(f)

    return w2i, vocabs["domain_vocab"], vocabs["tld_vocab"], vocabs["user_vocab"]


def load_regressor_model(regressor_ckpt: dict) -> QuantileRegressionModel:
    regressor_config = regressor_ckpt["config"]
    regressor = QuantileRegressionModel(**regressor_config)
    regressor.load_state_dict(regressor_ckpt["model_state_dict"])
    regressor.eval()
    return regressor


class RegressorModelPredictor:
    def __init__(self):

        self.w2i, self.domain_vocab, self.tld_vocab, self.user_vocab = (
            get_full_model_preprocessor()
        )
        self.columns, self.user_features, self.Tmin, self.Tmax, self.feature_columns = (
            load_user_data()
        )
        regressor_ckpt = torch.load(REGRESSOR_PATH, map_location="cpu")
        logging.info(f"Torch file has: {list(regressor_ckpt.keys())}")
        self.regressor = load_regressor_model(regressor_ckpt)
        self.quantile_probs = regressor_ckpt["quantile_probs"]
        sc = np.load(SCALER_PATH)
        self.feat_mean = torch.from_numpy(sc["mean"]).float().unsqueeze(0)  # [1, D]
        self.feat_std = torch.from_numpy(sc["std"]).float().unsqueeze(0)  # [1, D]
        self.clip_thr = float(sc["threshold"][0])

    def preprocess_input(self, data: dict):
        # Get user features from memory (instant lookup)
        username = data["by"]
        if username in self.user_features:
            row = self.user_features[username]
        else:
            # New user - all zeros
            row = {col: 0 for col in self.columns}

        row.pop("id", None)
        row["by"] = data["by"]
        row["title"] = data["title"]
        row["url"] = data["url"]
        row["time"] = data["time"]
        return row

    def get_tensors(self, input_data: dict):
        features_vec = self.preprocess_input(input_data)
        logging.info(f"Features vec: {features_vec}")
        data = hn_predict_utils.process_row(
            features_vec,
            self.Tmin,
            self.Tmax,
            self.feature_columns,
            self.w2i,
            self.domain_vocab,
            self.tld_vocab,
            self.user_vocab,
        )
        logging.info(f"Data: {data}")
        features_num = torch.tensor(
            data["features_num"], dtype=torch.float32
        ).unsqueeze(0)
        features_num = (features_num - self.feat_mean) / self.feat_std
        features_num.clamp_(-self.clip_thr, self.clip_thr)

        title_indices = torch.tensor(data["token_idx"], dtype=torch.long)
        if title_indices.numel() == 0:
            title_indices = torch.zeros(1, dtype=torch.long)  # ensure at least [0]
        title_indices = title_indices.unsqueeze(0)

        # Load categorical indices
        domain_indices = torch.tensor([data["domain_idx"]], dtype=torch.long)
        tld_indices = torch.tensor([data["tld_idx"]], dtype=torch.long)
        user_indices = torch.tensor([data["user_idx"]], dtype=torch.long)
        return features_num, title_indices, domain_indices, tld_indices, user_indices

    def predict(self, input_data: dict):
        logging.info("Getting tensors")
        features_num, title_indices, domain_indices, tld_indices, user_indices = (
            self.get_tensors(input_data)
        )
        self.regressor.eval()
        with torch.no_grad():
            logging.info("Calling model")
            reg_output = self.regressor(
                features_num, title_indices, domain_indices, tld_indices, user_indices
            )
            q_log = reg_output[0]
            q_log = torch.cummax(q_log, dim=0).values
            q_lin = torch.expm1(q_log)
            lo_idx = max(i for i, q in enumerate(self.quantile_probs) if q <= 0.5)
            hi_idx = min(i for i, q in enumerate(self.quantile_probs) if q >= 0.5)
            p_lo, p_hi = self.quantile_probs[lo_idx], self.quantile_probs[hi_idx]
            y_lo, y_hi = q_log[lo_idx], q_log[hi_idx]
            t = (0.5 - p_lo) / max(1e-6, (p_hi - p_lo))
            median_log = y_lo + t * (y_hi - y_lo)

            logging.info(
                "Output: num[min,max]=[%.2f, %.2f] | title_len=%d | q[min,max]=[%.1f, %.1f]",
                features_num.min().item(),
                features_num.max().item(),
                title_indices.size(1),
                q_lin.min().item(),
                q_lin.max().item(),
            )
            median = torch.expm1(median_log).item()
            return {"median": float(median), "quantiles": [float(x) for x in q_lin]}


def get_predictor():
    return RegressorModelPredictor()
