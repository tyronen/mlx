# pyright: reportCallIssue=false
import os
import hashlib

from datasets import load_dataset
from common import utils
from training.prepare_vocab import prepare_vocab
from common.utils import tokenize_text

COMMENT_SAMPLE = 8


def tokenize_title_batch(batch):
    types = batch["type"]
    titles = batch["title"]
    texts = batch["text"]
    out = []
    append = out.append
    for t, ttl, txt in zip(types, titles, texts):
        s = ttl if t == "story" else (txt if t == "comment" else None)
        append(tokenize_text(s) if isinstance(s, str) and s.strip() else [])

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
    out_dir = "data/hn_predict"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load minimal columns
    ds = load_dataset("OpenPipe/hacker-news", split="train")
    cols = ds.column_names or []
    keep = [c for c in ("type", "title", "text", "id") if c in cols]
    ds = ds.remove_columns([c for c in cols if c not in keep])
    ds = ds.filter(keep_row)

    # 2) Tokenize in parallel (cached on disk by HF)
    ds = ds.map(
        tokenize_title_batch,
        batched=True,
        batch_size=10_000,
        num_proc=os.cpu_count() or 4,
        desc="Tokenizing (batched)",
    )
    prepare_vocab(
        ds,
        min_freq=35,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
