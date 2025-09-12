import argparse
import html
import os
import re

from datasets import load_dataset
from tqdm import tqdm

from models import hn_predict

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/hn_corpus.txt")

URL_RE = re.compile(r"http[s]?://\S+")
HTML_RE = re.compile(r"<[^>]+>")
NT_RE = re.compile(r"n't\b")
APOS_RE = re.compile(r"'[a-z]*\b")
NON_KEEP_RE = re.compile(r"[^a-zA-Z0-9\s\-\!]")
NL_RE = re.compile(r"\s+")


def clean_text(text):
    """Clean and tokenize text similar to text8 format"""
    # Decode HTML entities first
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = URL_RE.sub(" ", text)

    # Remove HTML tags
    text = HTML_RE.sub(" ", text)

    # Remove apostrophe suffixes
    text = NT_RE.sub(" not", text)  # "don't" -> "do not"
    text = APOS_RE.sub("", text)

    # Keep only letters, numbers, and basic punctuation
    text = NON_KEEP_RE.sub(" ", text)

    # Replace multiple spaces/newlines with single space
    text = NL_RE.sub(" ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_and_clean_batch(batch):
    # batch: dict of lists
    types = batch["type"]
    titles = batch.get("title", [])
    texts = batch.get("text", [])
    out = []
    append = out.append
    for t, ti, te in zip(types, titles, texts):
        if t == "story":
            raw = ti or ""
        elif t == "comment":
            raw = te or ""
        else:
            continue
        if raw:
            cleaned = clean_text(raw)
            if cleaned:
                append(cleaned)
    return {"clean": out}


def iter_cleaned_nonstreaming(batch_size):
    """
    Load the HN dataset (non-streaming), clean in parallel with batched map,
    then iterate cleaned strings without materializing everything in RAM.
    """
    ds = load_dataset(hn_predict.DATASET_NAME, split="train")

    # keep only needed columns to cut copy cost
    keep = [c for c in ("type", "title", "text") if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    # parallel, batched clean -> a column "clean"
    ds = ds.map(
        extract_and_clean_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=(os.cpu_count() or 4),
        remove_columns=keep,  # drop originals; keep only "clean"
        desc="Cleaning in parallel",
    )

    # Iterate lazily from Arrow files (fast local reads, tiny Python overhead)
    for ex in tqdm(
        ds.to_iterable_dataset(num_shards=64),
        total=len(ds),  # now tqdm has a denominator
        unit="rec",
        ncols=100,
        desc="Emitting cleaned",
    ):
        s = ex["clean"]
        if s:
            yield s


if __name__ == "__main__":
    args = parser.parse_args()
    output_file = args.output

    FLUSH_BYTES = 4 * 1024 * 1024  # ~4MB chunks
    with open(output_file, "w", encoding="utf-8") as f:
        buf, buf_chars = [], 0
        for chunk in iter_cleaned_nonstreaming(batch_size=10000):
            buf.append(chunk)
            buf_chars += len(chunk) + 1
            if buf_chars >= FLUSH_BYTES:
                f.write(" ".join(buf))
                f.write(" ")
                buf.clear()
                buf_chars = 0
        if buf:
            f.write(" ".join(buf))
            f.write(" ")

    print("✅ Done")
