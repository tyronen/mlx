# type: ignore[reportPrivateImportUsage]
import json
import logging
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
from transformers import get_cosine_schedule_with_warmup
from models.msmarco_tokenizer import WORD2VEC_FILE
from .word2vec_prereq import WORD2VEC_DIR

from common import utils


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

EMBED_DIM = 400

hyperparameters = {
    "min_freq": 35,
    "context_size": 2,
    "embed_dim": EMBED_DIM,
    "epochs": 2,
    "learning_rate": 3e-3,
    "patience": 10000,
    "batch_size": 8192,
}

pair_count = 0


class SkipGramStream(IterableDataset):
    def __init__(self, indices, context_size):
        super().__init__()
        self.indices = indices
        self.context_size = context_size

    def __iter__(self):
        global pair_count
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
                    pair_count += 1
                    yield (center_word, context_word)

    def __len__(self):
        return None


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
    utils.setup_logging()
    logging.info(f"Using device: {device}")

    run = wandb.init(
        entity="tyronenicholas",
        project="Word2Vec",
        config=hyperparameters,
    )

    # === Build vocab ===
    pre = WORD2VEC_DIR
    idx_path = os.path.join(pre, "indices.int32.npy")
    counts_path = os.path.join(pre, "counts.int64.npy")
    word_to_ix_path = os.path.join(pre, "word_to_ix.json")
    assert (
        os.path.exists(idx_path)
        and os.path.exists(counts_path)
        and os.path.exists(word_to_ix_path)
    )

    indices = np.memmap(idx_path, dtype=np.int32, mode="r")
    counts = np.load(counts_path)
    with open(word_to_ix_path, "r", encoding="utf-8") as f:
        word_to_ix = json.load(f)
    ix_to_word = {int(v): k for k, v in word_to_ix.items()}  # for saving

    vocab_size = len(word_to_ix)
    logging.info(f"Vocab size: {vocab_size:,}  |  token stream: {len(indices):,}")
    # Add this after loading your truncated indices
    logging.info(f"Loaded indices shape: {indices.shape}")
    logging.info(f"First 10 indices: {indices[:10]}")
    logging.info(f"Last 10 indices: {indices[-10:]}")

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
    # model = torch.compile(model, mode="reduce-overhead")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=10000,  # Rough estimate is fine
        num_cycles=0.5,  # Decay to ~0 by end
    )
    _global_step = 0
    num_workers = 1
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
    scaler = torch.amp.GradScaler(device=device.type, enabled=True)

    epochs = hyperparameters["epochs"]
    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(data_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=100,
            unit="batch",
        )
        total_samples = 0
        epoch_loss = 0
        start_time = time.time()
        data_time = 0
        neg_time = 0
        forward_time = 0
        backward_time = 0
        total_time = 0
        back_time = 0
        step_time = 0
        for i, (center_batch, context_batch) in pbar:
            step_start = time.time()
            center_batch = center_batch.to(device, non_blocking=True)
            context_batch = context_batch.to(device, non_blocking=True)
            data_time += time.time() - step_start
            neg_start = time.time()
            neg_samples = get_negative_samples(probs, center_batch.size(0), 10)
            neg_time += time.time() - neg_start
            forward_start = time.time()
            with torch.autocast(device_type=device.type):
                loss = model(center_batch, context_batch, neg_samples)
            forward_time += time.time() - forward_start
            backward_start = time.time()
            scaler.scale(loss).backward()
            back_time += time.time() - backward_start
            opt_start = time.time()
            # Only step optimizer every N batches
            if i % 4 == 0:
                scaler.unscale_(optimizer)
                if i % 12 == 0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Don't zero gradients, let them accumulate
                pass
            # Only update scheduler every 10 steps
            if i % 10 == 0:
                for _ in range(10):  # Catch up the missed steps
                    scheduler.step()
                    _global_step += 1
            step_time += time.time() - opt_start
            backward_time += time.time() - backward_start

            total_samples += len(center_batch)
            epoch_loss += loss.item() * center_batch.size(0)
            total_time += time.time() - step_start
            if i % 100 == 1:
                run.log(
                    {
                        "train_loss_batch": loss.item(),
                        "grad_norm": float(total_grad_norm),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "data_time": data_time / i,
                        "neg_time": neg_time / i,
                        "forward_time": forward_time / i,
                        "backward_time": backward_time / i,
                        "back_time": back_time / i,
                        "step_time": step_time / i,
                        "total_time": total_time / i,
                        "pair_count": pair_count,
                    }
                )
            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "samples": total_samples,
                        "loss": f"{loss.item():.3f}",
                    }
                )
        logging.info("Finished the pbar")
        pure_training_time = time.time() - start_time
        samples_per_sec = total_samples / pure_training_time
        run.log(
            {
                "train_loss_epoch": epoch_loss / total_samples,
                "training_time": pure_training_time,
                "samples_per_sec": samples_per_sec,
            }
        )
        logging.info("Saving embeddings")
        torch.save(
            {
                "embeddings": model.in_embed.weight.data.cpu(),
                "word_to_ix": word_to_ix,
                "ix_to_word": ix_to_word,
            },
            WORD2VEC_FILE,
        )
        logging.info(f"✅ Embeddings saved to {WORD2VEC_FILE}")
    run.finish(0)


if __name__ == "__main__":
    main()
