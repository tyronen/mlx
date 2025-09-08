import argparse
import html
import re
import sys

from datasets import load_dataset, concatenate_datasets

from models import hn_predict

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/hn_corpus.txt")


def clean_text(text):
    """Clean and tokenize text similar to text8 format"""
    # Decode HTML entities first
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove apostrophe suffixes
    text = re.sub(r"n't\b", " not", text)  # "don't" -> "do not"
    text = re.sub(r"'[a-z]*\b", "", text)

    # Keep only letters, numbers, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\-\!]", " ", text)

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def iter_words_from_dataset():
    ds_dict = load_dataset(hn_predict.DATASET_NAME)
    splits = [ds_dict[k] for k in ds_dict.keys()]
    ds = splits[0] if len(splits) == 1 else concatenate_datasets(splits)
    for rec in ds:
        typ = rec.get("type")
        if typ == "story":
            txt = rec.get("title") or ""
        elif typ == "comment":
            txt = rec.get("text") or ""
        else:
            continue
        cleaned = clean_text(txt)
        if cleaned:
            for w in cleaned.split():
                yield w


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_file = sys.argv[1]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(iter_words_from_dataset()))
    print("✅ Done")
