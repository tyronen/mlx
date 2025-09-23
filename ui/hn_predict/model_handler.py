import json
import logging
import os
import pickle

import torch

from models import hn_predict_utils
from models.hn_predict import QuantileRegressionModel, CACHE_FILE

REGRESSOR_PATH = os.getenv("REGRESSION_MODEL_PATH", "data/best_quantile_epoch_4.pth")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def load_user_data():
    with open(CACHE_FILE, "rb") as fh:
        cache = pickle.load(fh)

    # propagate globals so hn_predict_utils.process_row has the right reference frame
    hn_predict_utils.global_Tmin = cache["global_Tmin"]
    hn_predict_utils.global_Tmax = cache["global_Tmax"]

    logging.info(
        f"Loaded inference cache with {len(cache['user_features'])} users "
        f"from {CACHE_FILE}"
    )
    return cache["columns"], cache["user_features"]


def get_full_model_preprocessor():
    """Load necessary data"""
    w2i, embedding_matrix = hn_predict_utils.load_embeddings(
        "skipgram_models/silvery200.pt"
    )
    hn_predict_utils.global_w2i = w2i
    hn_predict_utils.global_embedding_matrix = embedding_matrix
    # Load vocab sizes from vocab file
    with open(hn_predict_utils.TRAINING_VOCAB_PATH, "r") as f:
        vocabs = json.load(f)

    hn_predict_utils.global_domain_vocab = vocabs["domain_vocab"]
    hn_predict_utils.global_tld_vocab = vocabs["tld_vocab"]
    hn_predict_utils.global_user_vocab = vocabs["user_vocab"]


def load_regressor_model() -> QuantileRegressionModel:
    regressor_ckpt = torch.load(REGRESSOR_PATH, map_location="cpu")
    regressor_config = regressor_ckpt["config"]
    regressor = QuantileRegressionModel(**regressor_config)
    regressor.load_state_dict(regressor_ckpt["model_state_dict"])
    regressor.eval()
    return regressor


class RegressorModelPredictor:
    def __init__(self, regressor):
        self.columns, self.user_features = load_user_data()
        self.regressor = regressor

    def preprocess_input(self, data: dict) -> list[float]:
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
        print(features_vec)
        data = hn_predict_utils.process_row(features_vec)
        features_num = torch.tensor(
            data["features_num"], dtype=torch.float32
        ).unsqueeze(0)
        # Load title embeddings (precomputed)
        title_embeddings = torch.tensor(
            data["embedding"], dtype=torch.float32
        ).unsqueeze(0)

        # Load categorical indices
        domain_indices = torch.tensor([data["domain_idx"]], dtype=torch.long)
        tld_indices = torch.tensor([data["tld_idx"]], dtype=torch.long)
        user_indices = torch.tensor([data["user_idx"]], dtype=torch.long)
        return features_num, title_embeddings, domain_indices, tld_indices, user_indices

    def predict(self, input_data: dict) -> float:
        features_num, title_embeddings, domain_indices, tld_indices, user_indices = (
            self.get_tensors(input_data)
        )
        self.regressor.eval()
        with torch.no_grad():
            nonzero_mask = True

            features_num_nz = features_num[nonzero_mask]
            title_emb_nz = title_embeddings[nonzero_mask]
            domain_idx_nz = domain_indices[nonzero_mask]
            tld_idx_nz = tld_indices[nonzero_mask]
            user_idx_nz = user_indices[nonzero_mask]

            reg_output = self.regressor(
                features_num_nz, title_emb_nz, domain_idx_nz, tld_idx_nz, user_idx_nz
            )
            return reg_output[2].item()


def get_predictor():
    regressor = load_regressor_model()
    return RegressorModelPredictor(regressor)
