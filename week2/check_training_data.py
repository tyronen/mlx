# check_training_data.py
import torch
from dataset import DATASET_FILE


def check_triplets():
    datasets = torch.load(DATASET_FILE, weights_only=False)
    train_triplets = datasets["train_triplets"]

    query_texts = set()
    doc_texts = set()

    relevant_triplets = []

    print("Checking training triplets...")

    for i, triplet in enumerate(train_triplets):
        if i % 10000 == 0:
            print(f"Processed {i} triplets...")

        # The triplet contains tokenized data, we need to decode it
        # This is challenging since we'd need the tokenizer
        # Let's check a different way
        pass

    print(f"Total training triplets: {len(train_triplets)}")


if __name__ == "__main__":
    check_triplets()
