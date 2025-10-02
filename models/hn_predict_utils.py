# pyright: reportAttributeAccessIssue=false
import logging
from datetime import datetime, timezone
import pickle

import numpy as np
import pandas as pd
import tldextract

from common import utils

TRAINING_VOCAB_PATH = "data/train_vocab.json"
WORD_TO_IX_PATH = "data/word_to_ix.json"
POSTS_FILE = "data/posts.parquet"
SCALER_PATH = "data/scaler.npz"
CACHE_FILE = "data/inference_cache.pkl"

utils.setup_logging()


def load_user_data():
    with open(CACHE_FILE, "rb") as fh:
        cache = pickle.load(fh)

    # propagate globals so hn_predict_utils.process_row has the right reference frame
    feature_columns = [
        c
        for c in cache["columns"]
        if c not in ["id", "by", "time", "title", "score", "url", "num_posts"]
    ]

    logging.info(
        f"Loaded cache with {len(cache['user_features'])} users " f"from {CACHE_FILE}"
    )
    return (
        cache["columns"],
        cache["user_features"],
        cache["global_Tmin"],
        cache["global_Tmax"],
        feature_columns,
    )


def load_data(items_file, users_file):
    raw_items = pd.read_parquet(items_file)
    raw_users = pd.read_parquet(users_file)
    merged = pd.merge(raw_items, raw_users, on="by", how="left", suffixes=("", "_user"))
    has_score = merged.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[
        has_title["title"].str.strip().astype(bool)
    ]  # drop empty or whitespace-only
    return has_title.drop(columns=["id"])


def log_transform_plus1(x):
    if x <= 0:
        return x
    else:
        return np.log10(x + 1)


def time_transform(time):
    # Ensure epoch seconds are interpreted in UTC to match training
    if isinstance(time, (int, float)):
        timestamp = datetime.fromtimestamp(time, timezone.utc)
    else:
        timestamp = time

    year = timestamp.year
    hour_angle = 2 * np.pi * timestamp.hour / 24
    dow_angle = 2 * np.pi * timestamp.weekday() / 7
    day_angle = 2 * np.pi * (timestamp.timetuple().tm_yday - 1) / 365
    return year, hour_angle, dow_angle, day_angle


def extract_features(row, Tmin, Tmax, feature_columns):  # Time features
    year, hour_angle, dow_angle, day_angle = time_transform(row["time"])
    year_norm = (year - Tmin.year) / (Tmax.year - Tmin.year)

    time_feats = [
        year_norm,
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
        np.sin(day_angle),
        np.cos(day_angle),
        log_transform_plus1(row["num_posts"]),
    ]

    assert feature_columns is not None, "feature_columns not set"
    user_feats = [row.get(col, 0) for col in feature_columns]

    all_features = np.array(time_feats + user_feats, dtype=np.float32)
    return all_features


def process_row(
    row, Tmin, Tmax, feature_columns, w2i, domain_vocab, tld_vocab, user_vocab
):

    feats = extract_features(row, Tmin, Tmax, feature_columns)
    url = normalize_url(row["url"])
    domain = tldextract.extract(url).domain or ""
    tld = tldextract.extract(url).suffix or ""
    user = row["by"] or ""

    domain_idx = domain_vocab.get(domain, 0)
    tld_idx = tld_vocab.get(tld, 0)
    user_idx = user_vocab.get(user, 0)

    tokens = tokenize_title(w2i, row["title"])

    retval = {
        "features_num": feats,
        "token_idx": tokens,
        "domain_idx": domain_idx,
        "tld_idx": tld_idx,
        "user_idx": user_idx,
    }
    if "score" in row:
        retval["target"] = np.clip(row["score"], 0, None)
    return retval


def normalize_url(url):
    if url is None or not str(url).strip():
        return "http://empty"
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def tokenize_title(w2i, title_text):
    # Use the same tokenizer used to build the vocab
    tokens = utils.tokenize_text(title_text or "")
    token_indices = [w2i.get(token, 0) for token in tokens]
    return token_indices
