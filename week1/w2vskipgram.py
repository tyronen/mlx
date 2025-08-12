import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random
from tqdm import tqdm
import os
import json

# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
# Data parameters
DATA_FILE = "data/text8"
MIN_FREQ = 10  # Minimum word frequency to be included in vocabulary
VALIDATION_SPLIT = 0.05  # Percentage of data to use for validation

# Model parameters
EMBEDDING_DIM = 300  # Dimensionality of the word embeddings
WINDOW_SIZE = 2  # Max context window size (words on each side of the center word)
NEG_SAMPLES = 10  # Number of negative samples for each positive sample

# Training parameters
BATCH_SIZE = 512
LEARNING_RATE = 0.01
EPOCHS = 5
CHECKPOINT_FILE = "embeddings/word2vec_checkpoint.pt"
BEST_MODEL_FILE = "embeddings/word2vec_best_model.pt"
CHECKPOINT_STEP = 2000  # Save checkpoint every N steps
EVALUATION_STEP = 2000  # Evaluate model on validation set every N steps
EARLY_STOPPING_PATIENCE = 3  # Stop after N evaluations with no improvement

# =============================================================================
# Step 1: Data Preparation
# =============================================================================
print("Starting Step 1: Data Preparation...")


def preprocess(file_path, min_freq):
    """
    Reads text data, builds a vocabulary, and converts words to integer indices.
    """
    print("Reading and processing text data...")
    with open(file_path, "r") as f:
        text = f.read().split()

    print(f"Original corpus has {len(text)} tokens.")

    # Count word frequencies and filter out rare words
    word_counts = Counter(text)
    text = [word for word in text if word_counts[word] > min_freq]
    print(f"Corpus after filtering rare words has {len(text)} tokens.")

    # Build vocabulary
    vocab = Counter(text)
    int2word = {i: word for i, word in enumerate(vocab)}
    word2int = {word: i for i, word in int2word.items()}

    # Convert the entire text to integer indices
    int_words = [word2int[word] for word in text]

    return int_words, vocab, int2word, word2int


# Load and preprocess the data
int_words, vocab, int2word, word2int = preprocess(DATA_FILE, MIN_FREQ)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# --- Create Training and Validation Sets ---
split_idx = int(len(int_words) * (1 - VALIDATION_SPLIT))
train_words = int_words[:split_idx]
valid_words = int_words[split_idx:]
print(f"Training set size: {len(train_words)}, Validation set size: {len(valid_words)}")


class SkipGramDataset(Dataset):
    """
    Optimized, memory-efficient PyTorch Dataset for Word2Vec.
    """

    def __init__(self, text, window_size, vocab_size):
        self.text = text
        self.window_size = window_size
        self.vocab_size = vocab_size

        word_counts = Counter(text)
        freqs = np.zeros(vocab_size)
        for word_idx, count in word_counts.items():
            if word_idx < vocab_size:
                freqs[word_idx] = count

        freqs_pow = freqs**0.75
        self.word_dist = freqs_pow / np.sum(freqs_pow)
        self.word_dist_tensor = torch.from_numpy(self.word_dist).float()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        center_word = self.text[idx]
        dynamic_window = random.randint(1, self.window_size)
        start_idx = max(0, idx - dynamic_window)
        end_idx = min(len(self.text), idx + dynamic_window + 1)
        possible_context_indices = [i for i in range(start_idx, end_idx) if i != idx]

        context_word = self.text[
            random.choice(possible_context_indices) if possible_context_indices else idx
        ]
        neg_samples = torch.multinomial(
            self.word_dist_tensor, NEG_SAMPLES, replacement=True
        )

        return torch.tensor(center_word), torch.tensor(context_word), neg_samples


# Create datasets and dataloaders
train_dataset = SkipGramDataset(train_words, WINDOW_SIZE, vocab_size)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

valid_dataset = SkipGramDataset(valid_words, WINDOW_SIZE, vocab_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2)

print("Data Preparation complete.")

# =============================================================================
# Step 2: Model Definition
# =============================================================================
print("\nStarting Step 2: Model Definition...")


class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramNegativeSampling, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward(self, center_words, context_words, neg_samples):
        center_embeds = self.in_embed(center_words).unsqueeze(2)
        context_embeds = self.out_embed(context_words).unsqueeze(1)
        neg_embeds = self.out_embed(neg_samples)
        pos_scores = torch.bmm(context_embeds, center_embeds).squeeze(2)
        neg_scores = torch.bmm(neg_embeds, center_embeds).squeeze(2)
        return pos_scores, neg_scores


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        pos_loss = torch.log(torch.sigmoid(pos_scores))
        neg_loss = torch.log(torch.sigmoid(-neg_scores)).sum(1)
        return -(pos_loss + neg_loss).mean()


model = SkipGramNegativeSampling(vocab_size, EMBEDDING_DIM)
criterion = NegativeSamplingLoss()
print("Model Definition complete.")

# =============================================================================
# Step 3: Training
# =============================================================================
print("\nStarting Step 3: Training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_epoch = 0
global_step = 0
best_validation_loss = float("inf")
patience_counter = 0

if os.path.exists(CHECKPOINT_FILE):
    print(f"Resuming training from '{CHECKPOINT_FILE}'...")
    checkpoint = torch.load(CHECKPOINT_FILE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
    patience_counter = checkpoint.get("patience_counter", 0)

early_stop = False
for epoch in range(start_epoch, EPOCHS):
    model.train()
    progress_bar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        initial=global_step % len(train_dataloader),
    )

    for center_words, context_words, neg_samples in progress_bar:
        center_words, context_words, neg_samples = (
            center_words.to(device),
            context_words.to(device),
            neg_samples.to(device),
        )

        optimizer.zero_grad()
        pos_scores, neg_scores = model(center_words, context_words, neg_samples)
        loss = criterion(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        global_step += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        # --- Evaluation and Early Stopping ---
        if global_step % EVALUATION_STEP == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_center, val_context, val_neg in valid_dataloader:
                    val_center, val_context, val_neg = (
                        val_center.to(device),
                        val_context.to(device),
                        val_neg.to(device),
                    )
                    pos, neg = model(val_center, val_context, val_neg)
                    total_val_loss += criterion(pos, neg).item()

            avg_val_loss = total_val_loss / len(valid_dataloader)
            print(f"\nStep {global_step}: Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_validation_loss:
                best_validation_loss = avg_val_loss
                patience_counter = 0
                print(
                    f"New best validation loss. Saving best model to '{BEST_MODEL_FILE}'..."
                )
                torch.save({"model_state_dict": model.state_dict()}, BEST_MODEL_FILE)
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                early_stop = True
                print("Early stopping triggered.")
                break

            model.train()  # Switch back to training mode

        # --- Regular Checkpointing ---
        if global_step % CHECKPOINT_STEP == 0:
            print(f"\nSaving checkpoint at step {global_step}...")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation_loss": best_validation_loss,
                    "patience_counter": patience_counter,
                },
                CHECKPOINT_FILE,
            )

    if early_stop:
        break

print("Training complete.")

# =============================================================================
# Step 4: Save Final Embeddings
# =============================================================================
print("\nStarting Step 4: Saving Final Embeddings...")

# --- Load the best performing model before saving embeddings ---
if os.path.exists(BEST_MODEL_FILE):
    print(
        f"Loading best model from '{BEST_MODEL_FILE}' for final embedding extraction..."
    )
    best_model_cp = torch.load(BEST_MODEL_FILE)
    model.load_state_dict(best_model_cp["model_state_dict"])
xembeddings = model.in_embed.weight.cpu().data
torch.save(embeddings, "embeddings/skipgram_embeddings.pt")

with open("embeddings/word2int.json", "w") as f:
    json.dump(word2int, f)
with open("embeddings/int2word.json", "w") as f:
    json.dump(int2word, f)

print("Embeddings and vocabulary saved successfully.")
print("File 'embeddings/skipgram_embeddings.pt' created from the best model.")
