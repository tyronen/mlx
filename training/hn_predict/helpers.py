import numpy as np
import pandas as pd
import torch

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles,device):
        super().__init__()
        self.quantiles = torch.tensor(quantiles).float().view(1, -1).to(device)  # shape: (1, num_quantiles)

    def forward(self, preds, target):
        # preds: (batch_size, num_quantiles)
        # target: (batch_size, 1)
        target = target.unsqueeze(1)  # make sure shape is (batch_size, 1)
        errors = target - preds
        losses = torch.maximum(self.quantiles * errors, (self.quantiles - 1) * errors)
        return losses.mean()

def load_data(items_file, users_file):
    raw_items = pd.read_parquet(items_file)
    raw_users = pd.read_parquet(users_file)
    merged = pd.merge(raw_items, raw_users, on="by", how="left", suffixes=("", "_user"))
    has_score = merged.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[has_title["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    return has_title.drop(columns=["id"])


def load_embeddings(embeddings_file):
    edict = torch.load(embeddings_file, weights_only=True)
    embeddings = edict["embeddings"]
    word_to_ix = edict["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def log_transform_plus1(x):
    if x <= 0:
        return x
    else:
        return np.log10(x+1)

def time_transform(time):
    timestamp = pd.to_datetime(time)            
    year = timestamp.year  
    hour_angle = 2 * np.pi * timestamp.hour / 24
    dow_angle = 2 * np.pi * timestamp.dayofweek / 7
    day_angle = 2 * np.pi * timestamp.dayofyear / 365
    return year, hour_angle, dow_angle, day_angle
