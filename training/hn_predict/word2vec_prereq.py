#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import re

import numpy as np
from datasets import load_dataset

from common import utils

TOKEN_RE = re.compile(r"[A-Za-z0-9.+#_]+")
COMMENT_SAMPLE = 8


def tokenize_title_batch(batch):
    types = batch["type"]
    titles = batch["title"]
    texts = batch["text"]
    out = []
    append = out.append
    for t, ttl, txt in zip(types, titles, texts):
        s = ttl if t == "story" else (txt if t == "comment" else None)
        if isinstance(s, str) and s.strip():
            append(TOKEN_RE.findall(s.lower()))
        else:
            append([])
    return {"tokens": out}


def keep_row(ex):
    t = ex.get("type", None)
    if t == "story":
        return True
    if t != "comment":
        return False
    raw_id = ex.get("id", "")
    try:
        n = int(raw_id)
    except Exception:
        n = int(hashlib.md5(str(raw_id).encode("utf-8")).hexdigest(), 16)
    return (n % COMMENT_SAMPLE) == 0


def main():
    utils.setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--min_freq", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=10_000)
    ap.add_argument("--num_proc", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # 1) Load minimal columns
    ds = load_dataset("OpenPipe/hacker-news", split="train")
    keep = [c for c in ("type", "title", "text", "id") if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds = ds.filter(keep_row)

    # 2) Tokenize in parallel (cached on disk by HF)
    ds = ds.map(
        tokenize_title_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Tokenizing (batched)",
    )

    # 3) Pass 1: count tokens -> vocab  (PANDAS, VECTORIZED)
    # ds already has a "tokens" column (list[str]) from your batched map.
    # Convert to a DataFrame once, explode to a flat Series, and value_counts.

    df = ds.to_pandas()[["tokens"]]
    # explode handles [] by producing NaN; dropna removes them
    s = df["tokens"].explode(ignore_index=True).dropna()

    # counts for every token (unsorted to avoid extra work)
    vc = s.value_counts(sort=False)
    logging.info(f"Vocab size at min_freq=5: {(vc >= 5).sum()}")
    logging.info(f"Vocab size at min_freq=15: {(vc >= 15).sum()}")
    logging.info(f"Vocab size at min_freq=35: {(vc >= 35).sum()}")
    # vocab and mapping
    vc_kept = vc[vc >= args.min_freq]
    # if you want sorted vocab for reproducibility, uncomment:
    # vc_kept = vc_kept.sort_index(kind="mergesort")
    vocab = vc_kept.index.tolist()
    word_to_ix = dict(zip(vocab, range(len(vocab))))
    ix_to_word = {i: w for i, w in enumerate(vocab)}

    logging.info(f"Vocab ≥{args.min_freq}: {len(vocab):,}")

    # 4) Pass 2: total kept tokens (vectorized)
    # Filter the flat Series to only vocab, map to ids, and measure length
    idx_series = s[s.isin(word_to_ix)].map(word_to_ix).astype("int32")
    # Take every Nth token to maintain temporal distribution
    step = len(idx_series) // 10_000_000
    idx_series = idx_series.iloc[::step]
    total_kept = int(idx_series.shape[0])
    logging.info(f"Total kept tokens: {total_kept:,}")

    # 5) Pass 3: write indices (memmap) and counts array (aligned to vocab)
    idx_path = os.path.join(args.out_dir, "indices.int32.npy")
    counts_path = os.path.join(args.out_dir, "counts.int64.npy")
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    ix2_path = os.path.join(args.out_dir, "ix_to_word.json")

    # counts aligned to vocab order in O(1) using reindex
    counts = vc.reindex(vocab).fillna(0).to_numpy(dtype="int64")

    # write indices in one shot
    indices = np.memmap(idx_path, dtype=np.int32, mode="w+", shape=(total_kept,))
    indices[:] = idx_series.to_numpy(copy=False)
    indices.flush()
    np.save(counts_path, counts, allow_pickle=False)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word_to_ix, f, ensure_ascii=False)

    with open(ix2_path, "w", encoding="utf-8") as f:
        json.dump(ix_to_word, f, ensure_ascii=False)

    logging.info("✅ Wrote:")
    logging.info(idx_path)
    logging.info(counts_path)
    logging.info(vocab_path)
    logging.info(ix2_path)


if __name__ == "__main__":
    main()
