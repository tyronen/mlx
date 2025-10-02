# pyright: reportCallIssue=false
import os

from datasets import load_dataset
from common import utils
from training.prepare_vocab import prepare_vocab
from common.utils import tokenize_text

WORD2VEC_DIR = "data/msmarco_search"


def tokenize_batch(batch):
    # Process each row individually to maintain 1:1 mapping
    all_tokens = []

    for i in range(len(batch["query"])):
        row_tokens = []

        # Handle answers (list of strings)
        for answer in batch["answers"][i]:
            row_tokens.extend(tokenize_text(answer))

        # Handle query (single string)
        row_tokens.extend(tokenize_text(batch["query"][i]))

        # Handle passages (list of strings)
        for passage in batch["passages"][i]["passage_text"]:
            row_tokens.extend(tokenize_text(passage))

        all_tokens.append(row_tokens)

    return {"tokens": all_tokens}


def main():
    utils.setup_logging()
    os.makedirs(WORD2VEC_DIR, exist_ok=True)
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    ds = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=10_000,
        num_proc=os.cpu_count() or 4,
        desc="Tokenizing (batched)",
    )
    prepare_vocab(
        ds,
        min_freq=10,
        out_dir=WORD2VEC_DIR,
    )


if __name__ == "__main__":
    main()
