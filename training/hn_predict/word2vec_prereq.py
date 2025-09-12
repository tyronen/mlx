#!/usr/bin/env python3
import argparse
import json
import os
import re

import numpy as np
from datasets import load_dataset

TOKEN_RE = re.compile(r"[A-Za-z0-9.+#_]+")


def tokenize_title_batch(batch):
    types = batch["type"]
    titles = batch["title"]
    out = []
    append = out.append
    for t, s in zip(types, titles):
        if t == "story" and isinstance(s, str) and s.strip():
            append(TOKEN_RE.findall(s.lower()))
        else:
            append([])
    return {"tokens": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--min_freq", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=10_000)
    ap.add_argument("--num_proc", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # 1) Load minimal columns
    ds = load_dataset("OpenPipe/hacker-news", split="train")
    keep = [c for c in ("type", "title") if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    # 2) Tokenize in parallel (cached on disk by HF)
    ds = ds.map(
        tokenize_title_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="Tokenizing titles (batched)",
    )

    # 3) Pass 1: count tokens -> vocab  (PANDAS, VECTORIZED)
    # ds already has a "tokens" column (list[str]) from your batched map.
    # Convert to a DataFrame once, explode to a flat Series, and value_counts.

    df = ds.to_pandas()[["tokens"]]
    # explode handles [] by producing NaN; dropna removes them
    s = df["tokens"].explode(ignore_index=True).dropna()

    # counts for every token (unsorted to avoid extra work)
    vc = s.value_counts(sort=False)

    # vocab and mapping
    vc_kept = vc[vc >= args.min_freq]
    # if you want sorted vocab for reproducibility, uncomment:
    # vc_kept = vc_kept.sort_index(kind="mergesort")
    vocab = vc_kept.index.tolist()
    word_to_ix = dict(zip(vocab, range(len(vocab))))
    ix_to_word = {i: w for i, w in enumerate(vocab)}

    print(f"Vocab ≥{args.min_freq}: {len(vocab):,}")

    # 4) Pass 2: total kept tokens (vectorized)
    # Filter the flat Series to only vocab, map to ids, and measure length
    idx_series = s[s.isin(word_to_ix)].map(word_to_ix).astype("int32")
    total_kept = int(idx_series.shape[0])
    print(f"Total kept tokens: {total_kept:,}")

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

    print("✅ Wrote:")
    print("  ", idx_path)
    print("  ", counts_path)
    print("  ", vocab_path)
    print("  ", ix2_path)


if __name__ == "__main__":
    main()
