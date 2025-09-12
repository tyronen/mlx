import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from common import utils

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

hyperparameters = {
    "min_freq": 35,
    "context_size": 2,
    "embed_dim": 400,
    "epochs": 5,
    "learning_rate": 3e-3,
    "patience": 10000,
    "batch_size": 8192,
}

parser = argparse.ArgumentParser(
    description="Train skipgram word2vec model with negative sampling."
)
parser.add_argument(
    "--model",
    default="data/word2vec_skipgram.pth",
    help="Output file to save embeddings",
)
parser.add_argument(
    "--preproc_dir",
    default="data",
    help="Directory containing indices.int32.npy, counts.int64.npy, vocab.json",
)
args = parser.parse_args()

outfile = args.model


class SkipGramStream(IterableDataset):
    def __init__(self, indices, context_size):
        super().__init__()
        self.indices = indices
        self.context_size = context_size

    def __iter__(self):
        cs = self.context_size
        idxs = self.indices
        N = len(idxs)
        for i in range(cs, N - cs):
            center_word = int(idxs[i])
            # Dynamic window - randomly choose smaller windows
            dynamic_window = random.randint(1, cs)  # 1..context_size
            for j in range(i - dynamic_window, i + dynamic_window + 1):
                if j != i and 0 <= j < N:
                    context_word = int(idxs[j])
                    yield (center_word, context_word)

    def __len__(self):
        cs = self.context_size
        N = len(self.indices)
        if N <= 2 * cs:
            return 0
        # expected number of (center, context) pairs with dynamic window 1..cs
        return (N - 2 * cs) * (cs + 1)


# Add this to reduce very common words like "the", "and"
def subsample_frequent_words(indices, word_counts, threshold):
    total_words = sum(word_counts.values())
    subsampled = []
    for word_idx in tqdm(indices, desc="Subsampling", unit="tok", ncols=100):
        word_freq = word_counts[word_idx] / total_words
        prob = min(1.0, (threshold / word_freq) ** 0.5 + threshold / word_freq)
        if random.random() < prob:
            subsampled.append(word_idx)
    return subsampled


# === Model ===
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        embed_dim = hyperparameters["embed_dim"]
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.in_embed.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.out_embed.weight, mean=0.0, std=0.01)

    def forward(self, center_words, context_words, neg_samples):
        # Center word embeddings
        center_embeds = self.in_embed(center_words)  # [batch, embed_dim]

        # Context word embeddings
        context_embeds = self.out_embed(context_words)  # [batch, embed_dim]
        neg_embeds = self.out_embed(neg_samples)  # [batch, num_neg, embed_dim]

        # Dot products
        pos_scores = (center_embeds * context_embeds).sum(dim=1)  # [batch]
        neg_scores = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze(
            2
        )  # [batch, num_neg]

        # Loss
        pos_loss = F.logsigmoid(pos_scores)
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


def get_negative_samples(probs, bs, k):
    return torch.multinomial(probs, num_samples=bs * k, replacement=True).view(bs, k)


def main():
    device = utils.get_device()
    logging.info(f"Using device: {device}")

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="tyronenicholas",
        # Set the wandb project where this run will be logged.
        project="Word2Vec",
        # Track hyperparameters and run metadata.
        config=hyperparameters,
    )

    # === Build vocab ===
    pre = args.preproc_dir
    idx_path = os.path.join(pre, "indices.int32.npy")
    counts_path = os.path.join(pre, "counts.int64.npy")
    vocab_path = os.path.join(pre, "vocab.json")
    assert (
        os.path.exists(idx_path)
        and os.path.exists(counts_path)
        and os.path.exists(vocab_path)
    )

    indices = np.memmap(idx_path, dtype=np.int32, mode="r")
    counts = np.load(counts_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        word_to_ix = json.load(f)
    ix_to_word = {int(v): k for k, v in word_to_ix.items()}  # for saving

    vocab_size = len(word_to_ix)
    logging.info(f"Vocab size: {vocab_size:,}  |  token stream: {len(indices):,}")

    # Negatives distribution (unigram^0.75)
    cnt = counts.astype(np.float64)
    probs = torch.tensor(
        (cnt**0.75) / (cnt**0.75).sum(), dtype=torch.float, device=device
    )

    word2vec_dataset = SkipGramStream(
        indices=indices,
        context_size=hyperparameters["context_size"],
    )
    pin_memory = device.type == "cuda"

    model = SkipGramNegativeSampling(vocab_size).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    steps_per_epoch = math.ceil(len(word2vec_dataset) / hyperparameters["batch_size"])
    total_steps = steps_per_epoch * hyperparameters["epochs"]

    warmup_steps = max(
        1000, total_steps // 50
    )  # ~2% of training or 1k, whichever larger

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        # linear decay down to 10% of base LR
        remain = max(1, total_steps - warmup_steps)
        frac = 1.0 - (step - warmup_steps) / remain
        return max(0.10, frac)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    _global_step = 0
    num_workers = min(16, (os.cpu_count() or 8))
    data_loader = DataLoader(
        word2vec_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=8,
        multiprocessing_context="forkserver",
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=False)

    epochs = hyperparameters["epochs"]
    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=100,
            unit="batch",
        )
        total_samples = 0
        epoch_loss = 0
        start_time = time.time()
        for i, (center_batch, context_batch) in pbar:
            center_batch = center_batch.to(device, non_blocking=True)
            context_batch = context_batch.to(device, non_blocking=True)
            neg_samples = get_negative_samples(probs, center_batch.size(0), 10)

            optimizer.zero_grad()
            if device.type == "mps":
                loss = model(center_batch, context_batch, neg_samples)  # Swapped order
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
            else:
                with torch.autocast(device_type=device.type):
                    loss = model(center_batch, context_batch, neg_samples)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                _global_step += 1
                scheduler.step()

            total_samples += len(center_batch)
            epoch_loss += loss.item() * center_batch.size(0)
            if i % 100 == 0:
                run.log(
                    {
                        "train_loss_batch": loss.item(),
                        "grad_norm": float(total_grad_norm),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "samples": total_samples,
                        "loss": f"{loss.item():.3f}",
                    }
                )
        pure_training_time = time.time() - start_time
        samples_per_sec = total_samples / pure_training_time
        run.log(
            {
                "train_loss_epoch": epoch_loss / total_samples,
                "training_time": pure_training_time,
                "samples_per_sec": samples_per_sec,
            }
        )
        E_in = model.in_embed.weight.data.cpu()
        E_out = model.out_embed.weight.data.cpu()
        E_avg = (E_in + E_out) / 2
        E_avg = E_avg / (E_avg.norm(dim=1, keepdim=True) + 1e-9)

        torch.save(
            {"embeddings": E_avg, "word_to_ix": word_to_ix, "ix_to_word": ix_to_word},
            outfile,
        )
    logging.info(f"✅ Embeddings saved to {outfile}")
    run.finish(0)


if __name__ == "__main__":
    main()
