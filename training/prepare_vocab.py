# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false

import json
import logging
import os

import numpy as np


def prepare_vocab(ds, min_freq, out_dir):
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
    vc_kept = vc[vc >= min_freq]
    # if you want sorted vocab for reproducibility, uncomment:
    # vc_kept = vc_kept.sort_index(kind="mergesort")
    vocab = vc_kept.index.tolist()
    word_to_ix = dict(zip(vocab, range(len(vocab))))
    ix_to_word = {i: w for i, w in enumerate(vocab)}

    logging.info(f"Vocab ≥{min_freq}: {len(vocab):,}")

    # 4) Pass 2: total kept tokens (vectorized)
    # Filter the flat Series to only vocab, map to ids, and measure length
    idx_series = s[s.isin(word_to_ix)].map(word_to_ix).astype("int32")
    # Take every Nth token to maintain temporal distribution
    step = len(idx_series) // 10_000_000
    idx_series = idx_series.iloc[::step]
    total_kept = int(idx_series.shape[0])
    logging.info(f"Total kept tokens: {total_kept:,}")

    # 5) Pass 3: write indices (memmap) and counts array (aligned to vocab)
    idx_path = os.path.join(out_dir, "indices.int32.npy")
    counts_path = os.path.join(out_dir, "counts.int64.npy")
    ix2_path = os.path.join(out_dir, "ix_to_word.json")
    w2i_path = os.path.join(out_dir, "word_to_ix.json")

    # counts aligned to vocab order in O(1) using reindex
    counts = vc.reindex(vocab).fillna(0).to_numpy(dtype="int64")

    # write indices in one shot
    indices = np.memmap(idx_path, dtype=np.int32, mode="w+", shape=(total_kept,))
    indices[:] = idx_series.to_numpy(copy=False)
    indices.flush()
    np.save(counts_path, counts, allow_pickle=False)

    with open(w2i_path, "w", encoding="utf-8") as f:
        json.dump(word_to_ix, f, ensure_ascii=False)

    with open(ix2_path, "w", encoding="utf-8") as f:
        json.dump(ix_to_word, f, ensure_ascii=False)

    logging.info("✅ Wrote:")
    logging.info(idx_path)
    logging.info(counts_path)
    logging.info(w2i_path)
    logging.info(ix2_path)
