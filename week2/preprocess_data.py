import logging
from datasets import load_dataset
import torch
from tqdm import tqdm

from dataset import TripletDataset, DATASET_FILE
import utils
from tokenizer import Word2VecTokenizer, MAX_LENGTH

from collections import Counter
from datasets import load_dataset


def build_doc_freq(ms_marco_data):
    """
    Returns:
        df (Counter):  word → document-frequency  (how many docs contain the word at least once)
        N            : total number of documents scanned
    """
    df = Counter()
    N = 0

    for split in ("train", "validation", "test"):
        for row in tqdm(ms_marco_data[split]):
            # Each passage counts as ONE document
            for text in row["passages"]["passage_text"]:
                N += 1
                # use a set so each word counted once per doc
                unique_words = set(text.lower().split()[:MAX_LENGTH])
                df.update(unique_words)

    return df, N


def main():
    utils.setup_logging()
    device = utils.get_device()

    logging.info("Loading MS MARCO dataset...")
    ms_marco_data = load_dataset("ms_marco", "v1.1")

    logging.info("Building document frequency...")
    df, N = build_doc_freq(ms_marco_data)

    tokenizer = Word2VecTokenizer(doc_freq=df)

    logging.info("Creating training dataset...")
    train_dataset = TripletDataset(ms_marco_data["train"], tokenizer, device)

    logging.info("Creating validation dataset...")
    validation_dataset = TripletDataset(ms_marco_data["validation"], tokenizer, device)

    logging.info("Creating test dataset...")
    test_dataset = TripletDataset(ms_marco_data["test"], tokenizer, device)

    logging.info("Saving preprocessed datasets...")
    torch.save(
        {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
            "tokenizer": tokenizer,
            "document_frequency": df,
            "num_passages": N,
        },
        DATASET_FILE,
    )

    logging.info(
        f"Saved datasets with sizes: train={len(train_dataset)}, val={len(validation_dataset)}, test={len(test_dataset)}"
    )


if __name__ == "__main__":
    main()
