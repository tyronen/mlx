import gc
import json
import logging
import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd
import tldextract
import torch
from tqdm import tqdm

from common import utils
from common.utils import tokenize_text
from models.hn_predict_utils import load_embeddings, log_transform_plus1, time_transform

# global variables to be shared across workers
global_w2i = None
global_Tmin = None
global_Tmax = None
global_domain_vocab = None
global_tld_vocab = None
global_user_vocab = None

UNK_TOKEN = "<unk>"


def init_worker(w2i_dict, Tmin, Tmax, domain_vocab, tld_vocab, user_vocab):
    global global_w2i
    global global_Tmin
    global global_Tmax
    global global_domain_vocab
    global global_tld_vocab
    global global_user_vocab

    global_w2i = w2i_dict
    global_Tmin = Tmin
    global_Tmax = Tmax
    global_domain_vocab = domain_vocab
    global_tld_vocab = tld_vocab
    global_user_vocab = user_vocab


def build_vocab(values, min_freq=1, topk=None):
    counter = Counter(values)
    items = [(v, count) for v, count in counter.items() if count >= min_freq]
    if topk is not None:
        items = sorted(items, key=lambda x: -x[1])[:topk]

    vocab = {UNK_TOKEN: 0}  # reserve index 0 for unknown token
    for idx, (v, count) in enumerate(items, start=1):
        vocab[v] = idx
    return vocab


def extract_features(row):
    # Time features
    year, hour_angle, dow_angle, day_angle = time_transform(row["time"])
    year_norm = (year - global_Tmin.year) / (global_Tmax.year - global_Tmin.year)

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

    # Collect all remaining user features (everything except those handled above
    user_feature_names = [
        col
        for col in row.keys()
        if col not in ["by", "time", "title", "score", "url", "num_posts"]
    ]

    user_feats = [row[col] for col in user_feature_names]

    all_features = np.array(time_feats + user_feats, dtype=np.float32)
    return all_features


def normalize_url(url):
    if url is None or not str(url).strip():
        return "http://empty"
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def tokenize_title(title_text):
    tokens = tokenize_text(title_text)
    token_indices = [global_w2i.get(token, 0) for token in tokens]
    return token_indices


def process_row(row):

    feats = extract_features(row)
    url = normalize_url(row["url"])
    domain = tldextract.extract(url).domain or ""
    tld = tldextract.extract(url).suffix or ""
    user = row["by"] or ""

    domain_idx = global_domain_vocab.get(domain, 0)
    tld_idx = global_tld_vocab.get(tld, 0)
    user_idx = global_user_vocab.get(user, 0)

    tokens = tokenize_title(row["title"])
    target = np.clip(row["score"], 0, None)

    return {
        "features_num": feats,
        "token_idx": tokens,
        "domain_idx": domain_idx,
        "tld_idx": tld_idx,
        "user_idx": user_idx,
        "target": target,
    }


def prepare_for_16(array32, threshold):
    feat_mean = array32.mean(axis=0, dtype=np.float64)
    feat_std = array32.std(axis=0, dtype=np.float64)
    feat_std[feat_std < 1e-6] = 1e-6

    # z-score + clip to fp16-friendly range
    array32 = (array32 - feat_mean) / feat_std
    np.clip(array32, -threshold, threshold, out=array32)
    return array32


def precompute_parallel(
    df,
    w2i_dict,
    domain_vocab,
    tld_vocab,
    user_vocab,
    Tmin=None,
    Tmax=None,
    ref_time=None,
    compute_delta_t=False,
    num_workers=None,
):
    df = df.reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    if compute_delta_t:
        delta_t = (ref_time - df["time"]) / np.timedelta64(30, "D")
        delta_t = torch.tensor(delta_t, dtype=torch.float16)
    else:
        delta_t = None

    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            w2i_dict,
            Tmin,
            Tmax,
            domain_vocab,
            tld_vocab,
            user_vocab,
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_row, df.to_dict(orient="records")),
                total=len(df),
                desc="Precomputing",
            )
        )

    features_array = np.stack([r["features_num"] for r in results])
    all_token_idx = [r["token_idx"] for r in results]
    all_targets = torch.tensor([r["target"] for r in results], dtype=torch.float32)
    all_domain_idx = torch.tensor([r["domain_idx"] for r in results], dtype=torch.long)
    all_tld_idx = torch.tensor([r["tld_idx"] for r in results], dtype=torch.long)
    all_user_idx = torch.tensor([r["user_idx"] for r in results], dtype=torch.long)
    del results
    gc.collect()

    features_array = prepare_for_16(features_array, 25)

    all_features_num = torch.from_numpy(features_array).to(torch.float16)

    return (
        all_features_num,
        all_token_idx,
        all_domain_idx,
        all_tld_idx,
        all_user_idx,
        all_targets,
        delta_t,
    )


if __name__ == "__main__":
    utils.setup_logging()
    EMBEDDING_FILE = "data/word2vec_skipgram.pth"
    TRAINING_VOCAB_PATH = "data/train_vocab.json"
    FILEPATH = "data/posts.parquet"
    OUTPUT_DIR = "data"
    NUM_WORKERS = min(8, os.cpu_count() or 8)

    w2i = load_embeddings(EMBEDDING_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(FILEPATH)
    df = df.drop(["id"], axis=1)
    df = df.sort_values(by="time").reset_index(drop=True)
    df = df.dropna()

    train_size = int(0.9 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    T_ref = train_df["time"].max()
    Tmin = train_df["time"].min()
    Tmax = train_df["time"].max()

    domains = [
        tldextract.extract(normalize_url(url)).domain or "" for url in train_df["url"]
    ]
    tlds = [
        tldextract.extract(normalize_url(url)).suffix or "" for url in train_df["url"]
    ]
    users = train_df["by"].fillna("")

    domain_vocab = build_vocab(domains)
    tld_vocab = build_vocab(tlds, topk=5)
    user_vocab = build_vocab(users)

    # save vocab with unknown token
    all_vocabs = {
        "domain_vocab": domain_vocab,
        "tld_vocab": tld_vocab,
        "user_vocab": user_vocab,
    }

    with open(TRAINING_VOCAB_PATH, "w") as f:
        json.dump(all_vocabs, f)

    (
        train_features_num,
        train_title_idx,
        train_domain_idx,
        train_tld_idx,
        train_user_idx,
        train_targets,
        train_delta_t,
    ) = precompute_parallel(
        train_df,
        w2i,
        domain_vocab,
        tld_vocab,
        user_vocab,
        Tmin=Tmin,
        Tmax=Tmax,
        ref_time=T_ref,
        compute_delta_t=True,
        num_workers=NUM_WORKERS,
    )

    oov_counter = Counter()
    vocab = set(w2i.keys())

    for title in train_df["title"]:
        toks = tokenize_text(title)
        oov_counter.update([t for t in toks if t not in vocab])

    total_tokens = sum(len(tokenize_text(t)) for t in train_df["title"])
    oov_tokens = sum(oov_counter.values())
    logging.info(
        f"OOV token rate: {oov_tokens / total_tokens:.3%} over {total_tokens:,} tokens"
    )
    logging.info(
        "Top OOV tokens: "
        + ", ".join(f"{w}:{c}" for w, c in oov_counter.most_common(20))
    )

    torch.save(
        {
            "features_num": train_features_num,
            "title_index": train_title_idx,
            "domain_index": train_domain_idx,
            "tld_index": train_tld_idx,
            "user_index": train_user_idx,
            "delta_t": train_delta_t,
            "targets": train_targets,
        },
        os.path.join(OUTPUT_DIR, "train.pt"),
    )

    (
        val_features_num,
        val_title_idx,
        val_domain_idx,
        val_tld_idx,
        val_user_idx,
        val_targets,
        val_delta_t,
    ) = precompute_parallel(
        val_df,
        w2i,
        domain_vocab,
        tld_vocab,
        user_vocab,
        Tmin=Tmin,
        Tmax=Tmax,
        compute_delta_t=False,
        num_workers=NUM_WORKERS,
    )

    torch.save(
        {
            "features_num": val_features_num,
            "title_index": val_title_idx,
            "domain_index": val_domain_idx,
            "tld_index": val_tld_idx,
            "user_index": val_user_idx,
            "delta_t": val_delta_t,
            "targets": val_targets,
        },
        os.path.join(OUTPUT_DIR, "val.pt"),
    )

    logging.info("Precomputation finished!")
