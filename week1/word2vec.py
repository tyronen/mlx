import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import logging
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import argparse

min_freq = 10
context_size = 2
embed_dim = 300
batch_size = 512
epochs = 5
learning_rate = 0.01
patience = 10000

parser = argparse.ArgumentParser(description="Train CBOW word2vec model with negative sampling.")
parser.add_argument('--corpus', required=True, help='Input text file for training')
parser.add_argument('--model', required=True, help='Output file to save embeddings')
args = parser.parse_args()

input_file = args.corpus
outfile = args.model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)

# Auto-select device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class CBOWDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# === Build CBOW dataset ===
def make_cbow_dataset(indices):
    dataset = []
    for i in range(context_size, len(indices) - context_size):
        context = (
            indices[i - context_size : i] + indices[i + 1 : i + context_size + 1]
        )
        target = indices[i]
        dataset.append((context, target))
    return dataset

# === Model ===
class CBOWNegativeSampling(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # Separate output embeddings

    def forward(self, context, target, neg_samples):
        # Context embedding (average)
        context_embeds = self.in_embed(context).mean(dim=1)  # [batch, embed_dim]

        # Only compute for target + negatives (not all 47K words!)
        target_embeds = self.out_embed(target)  # [batch, embed_dim]
        neg_embeds = self.out_embed(neg_samples)  # [batch, num_neg, embed_dim]

        # Dot products (much faster than full linear layer)
        pos_scores = (context_embeds * target_embeds).sum(dim=1)  # [batch]
        neg_scores = torch.bmm(neg_embeds, context_embeds.unsqueeze(2)).squeeze(2)  # [batch, num_neg]

        # Compute loss
        pos_loss = F.logsigmoid(pos_scores)
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)

        return -(pos_loss + neg_loss).mean()

def get_negative_samples(target_batch, vocab_size, num_negative=10):
    samp_batch_size = target_batch.size(0)
    return torch.randint(0, vocab_size, (samp_batch_size, num_negative), device=target_batch.device)


def main():
    device = get_device()
    logging.info(f"Using device: {device}")

    # === Load ===
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().split()

    # === Build vocab ===
    counter = Counter(text)
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word_to_ix = {word: i for i, word in enumerate(sorted(vocab))}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    vocab_size = len(word_to_ix)

    logging.info(f"Vocab size: {vocab_size}")

    # === Convert text to indices ===
    indices = [word_to_ix[word] for word in text if word in word_to_ix]

    dataset = make_cbow_dataset(indices)
    cbow_dataset = CBOWDataset(dataset)
    pin_memory = device.type == "cuda"
    data_loader = DataLoader(
        cbow_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory
    )

    logging.info(f"Dataset size: {len(dataset)}")

    model = CBOWNegativeSampling(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    no_improve_count = 0
    for epoch in range(epochs):
        for i, (context_batch, target_batch) in enumerate(data_loader):
            context_batch = context_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            neg_samples = get_negative_samples(target_batch, vocab_size, num_negative=10)
            optimizer.zero_grad()
            loss = model(context_batch, target_batch, neg_samples)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improve_count = 0
                torch.save(
                    {
                        "embeddings": model.in_embed.weight.data.cpu(),
                        "word_to_ix": word_to_ix,
                        "ix_to_word": ix_to_word,
                    },
                    outfile,
                )
            else:
                no_improve_count += 1
            if no_improve_count > patience:
                logging.info(f"Early stopping at epoch {epoch + 1}, step {i}")
                break
            if i % 1000 == 999:
                logging.info(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}")
    logging.info(f"✅ Embeddings saved to {outfile}")

if __name__ == "__main__":
    main()