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
from models.hn_predict_utils import (
    VOCAB_PATH,
    POSTS_FILE,
    SCALER_PATH,
    TRAINING_VOCAB_PATH,
    load_user_data,
    process_row,
)

# global variables to be shared across workers
global_w2i = None
global_Tmin = None
global_Tmax = None
global_domain_vocab = None
global_tld_vocab = None
global_user_vocab = None
global_feature_columns = None

UNK_TOKEN = "<unk>"


def init_worker(
    w2i_dict, Tmin, Tmax, domain_vocab, tld_vocab, user_vocab, feature_columns
):
    global global_w2i
    global global_Tmin
    global global_Tmax
    global global_domain_vocab
    global global_tld_vocab
    global global_user_vocab
    global global_feature_columns

    global_w2i = w2i_dict
    global_Tmin = Tmin
    global_Tmax = Tmax
    global_domain_vocab = domain_vocab
    global_tld_vocab = tld_vocab
    global_user_vocab = user_vocab
    global_feature_columns = feature_columns


def build_vocab(values, min_freq=1, topk=None):
    counter = Counter(values)
    items = [(v, count) for v, count in counter.items() if count >= min_freq]
    if topk is not None:
        items = sorted(items, key=lambda x: -x[1])[:topk]

    vocab = {UNK_TOKEN: 0}  # reserve index 0 for unknown token
    for idx, (v, count) in enumerate(items, start=1):
        vocab[v] = idx
    return vocab


def normalize_url(url):
    if url is None or not str(url).strip():
        return "http://empty"
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def mp_row(row):
    return process_row(
        row,
        global_Tmin,
        global_Tmax,
        global_feature_columns,
        global_w2i,
        global_domain_vocab,
        global_tld_vocab,
        global_user_vocab,
    )


def prepare_for_16(array32, threshold):
    feat_mean = array32.mean(axis=0, dtype=np.float64)
    feat_std = array32.std(axis=0, dtype=np.float64)
    feat_std[feat_std < 1e-6] = 1e-6

    # z-score + clip to fp16-friendly range
    array32 = (array32 - feat_mean) / feat_std
    np.clip(array32, -threshold, threshold, out=array32)

    np.savez(
        SCALER_PATH,
        mean=feat_mean.astype(np.float32),
        std=feat_std.astype(np.float32),
        threshold=np.array([threshold], dtype=np.float32),
    )
    return array32


def precompute_parallel(
    df,
    w2i_dict,
    domain_vocab,
    tld_vocab,
    user_vocab,
    feature_columns,
    Tmin,
    Tmax,
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
            feature_columns,
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(mp_row, df.to_dict(orient="records")),
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


def main():
    utils.setup_logging()
    OUTPUT_DIR = "data"
    NUM_WORKERS = min(8, os.cpu_count() or 8)

    with open(VOCAB_PATH, "r") as f:
        w2i = json.load(f)
    if "UNK" not in w2i:
        w2i["UNK"] = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, _, Tmin, Tmax, feature_columns = load_user_data()

    df = pd.read_parquet(POSTS_FILE)
    df = df.drop(["id"], axis=1)
    df = df.sort_values(by="time").reset_index(drop=True)
    df = df.dropna()

    train_size = int(0.9 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    T_ref = train_df["time"].max()

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
        feature_columns,
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
        feature_columns,
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


if __name__ == "__main__":
    main()
