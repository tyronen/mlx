import pickle

import torch

from big_model.inference_preprocess import CACHE_FILE
from big_model.model import FullModel
from big_model import utils
import json
import logging
import os
from double_model.model import QuantileRegressionModel, ClassifierModel

FULL_MODEL_PATH = os.getenv("FULL_MODEL_PATH", "models/20250613_130611/best_model_5.pth")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "models/20250613_130611/best_model_1.pth")
REGRESSOR_PATH = os.getenv("REGRESSION_MODEL_PATH", "models/20250613_130611/best_model_2.pth")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

def load_user_data():
    with open(CACHE_FILE, "rb") as fh:
        cache = pickle.load(fh)

    # propagate globals so utils.process_row has the right reference frame
    utils.global_Tmin = cache["global_Tmin"]
    utils.global_Tmax = cache["global_Tmax"]

    logging.info(
        f"Loaded inference cache with {len(cache['user_features'])} users "
        f"from {CACHE_FILE}"
    )
    return cache["columns"], cache["user_features"]


def get_full_model_preprocessor():
    """Load necessary data"""
    w2i, embedding_matrix = utils.load_embeddings("skipgram_models/silvery200.pt")
    utils.global_w2i = w2i
    utils.global_embedding_matrix = embedding_matrix
    # Load vocab sizes from vocab file
    with open(utils.TRAINING_VOCAB_PATH, 'r') as f:
        vocabs = json.load(f)
    
    utils.global_domain_vocab = vocabs['domain_vocab']
    utils.global_tld_vocab = vocabs['tld_vocab']
    utils.global_user_vocab = vocabs['user_vocab']

def load_full_model() -> FullModel:
    """Load a FullModel from checkpoint"""
    device = utils.get_device()

    # Create model with correct parameters
    checkpoint = torch.load(FULL_MODEL_PATH, map_location=device)
    config = checkpoint["config"]
    model = FullModel(**config)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def load_double_model() -> tuple[ClassifierModel, QuantileRegressionModel]:
    """Load a FullModel from checkpoint"""
    device = utils.get_device()
    classifier_ckpt = torch.load(CLASSIFIER_PATH)
    regressor_ckpt = torch.load(REGRESSOR_PATH)
    classifier_config = classifier_ckpt["config"]
    regressor_config = regressor_ckpt["config"]
    classifier = ClassifierModel(**classifier_config)
    regressor = QuantileRegressionModel(**regressor_config)
    classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    regressor.load_state_dict(regressor_ckpt['model_state_dict'])
    classifier.eval()
    regressor.eval()
    return classifier, regressor


class BasePredictor:
    def __init__(self):
        self.columns, self.user_features = load_user_data()

    def preprocess_input(self, data: dict) -> list[float]:
        # Get user features from memory (instant lookup)
        username = data['by']
        if username in self.user_features:
            row = self.user_features[username]
        else:
            # New user - all zeros
            row = {col: 0 for col in self.columns}

        row.pop('id', None)
        row['by'] = data['by']
        row['title'] = data['title']
        row['url'] = data['url']
        row['time'] = data['time']
        return row

    def get_tensors(self, input_data:dict):
        features_vec = self.preprocess_input(input_data)
        print(features_vec)
        data = utils.process_row(features_vec)
        features_num = torch.tensor(data['features_num'], dtype=torch.float32).unsqueeze(0)
        # Load title embeddings (precomputed)
        title_embeddings = torch.tensor(data['embedding'], dtype=torch.float32).unsqueeze(0)

        # Load categorical indices
        domain_indices = torch.tensor([data['domain_idx']], dtype=torch.long)
        tld_indices = torch.tensor([data['tld_idx']], dtype=torch.long)
        user_indices = torch.tensor([data['user_idx']], dtype=torch.long)
        return features_num, title_embeddings, domain_indices, tld_indices, user_indices

class FullModelPredictor(BasePredictor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict(self, input_data: dict) -> float:
        inputs = self.get_tensors(input_data)
        #May need to modify if model is not preprocessed
        self.model.eval()
        with torch.no_grad():
            raw_prediction = self.model(*inputs)
            prediction = 10 ** raw_prediction.item() - 1
        #self.analyze_feature_importance(data)
        print(f"Final prediction: {prediction}")

        return prediction

    def analyze_feature_importance(self, data, top_k=10):
        features_num = torch.tensor(data['features_num'], dtype=torch.float32, requires_grad=True).unsqueeze(0)
        # Load title embeddings (precomputed)
        title_embeddings = torch.tensor(data['embedding'], dtype=torch.float32, requires_grad=True).unsqueeze(0)

        # Load categorical indices
        domain_indices = torch.tensor([data['domain_idx']], dtype=torch.long)
        tld_indices = torch.tensor([data['tld_idx']], dtype=torch.long)
        user_indices = torch.tensor([data['user_idx']], dtype=torch.long)
        # Clear any existing gradients
        if features_num.grad is not None:
            features_num.grad.zero_()
        if title_embeddings.grad is not None:
            title_embeddings.grad.zero_()

        features_num.retain_grad()
        title_embeddings.retain_grad()

        self.model.eval()

        # Forward pass WITH gradient computation
        raw_prediction = self.model(features_num, title_embeddings, domain_indices, tld_indices, user_indices)

        # Backward pass
        raw_prediction.backward()

        # Check gradients exist
        if features_num.grad is not None and title_embeddings.grad is not None:
            num_importance = torch.abs(features_num.grad).squeeze().numpy()
            title_importance = torch.abs(title_embeddings.grad).squeeze().numpy()

            importance_pairs = list(zip(self.columns[:len(num_importance)], num_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)

            print(f"Top {top_k} most important features:")
            for name, importance in importance_pairs[:top_k]:
                print(f"{name}: {importance:.4f}")

            print(f"\nTitle embedding importance (mean): {title_importance.mean():.4f}")
        else:
            print("Failed to compute gradients for feature importance")

class DoubleModelPredictor(BasePredictor):
    def __init__(self, classifier, regressor):
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor

    def predict(self, input_data: dict) -> float:
        features_num, title_embeddings, domain_indices, tld_indices, user_indices = self.get_tensors(input_data)
        self.classifier.eval()
        self.regressor.eval()
        with torch.no_grad():
            probs = self.classifier(features_num, title_embeddings, domain_indices, tld_indices, user_indices).squeeze()

            # If classifier predicts non-zero → run regressor
            if probs.item() <= 0.5:
                return 1
            nonzero_mask = True

            features_num_nz = features_num[nonzero_mask]
            title_emb_nz = title_embeddings[nonzero_mask]
            domain_idx_nz = domain_indices[nonzero_mask]
            tld_idx_nz = tld_indices[nonzero_mask]
            user_idx_nz = user_indices[nonzero_mask]

            reg_output = self.regressor(features_num_nz, title_emb_nz, domain_idx_nz, tld_idx_nz, user_idx_nz)
            return reg_output[2].item()


def get_predictor(model_type: str):
    if (model_type == "full_model"):
        model = load_full_model()
        get_full_model_preprocessor()
        return FullModelPredictor(model)
    classifer, regressor = load_double_model()
    return DoubleModelPredictor(classifer, regressor)
