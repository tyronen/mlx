# verify_model.py
#
# A script to verify if the trained models have collapsed by checking
# if they produce the same embedding for different input texts.

import torch
import numpy as np

# Import your project's utility and model files
import utils
from model import QueryTower, DocTower
from tokenizer import Word2VecTokenizer

print("--- Verification Script Started ---")

# --- 1. Setup and Load Models ---
# This section mirrors the setup in your FastAPI app and storage script.
print("--> Loading models and tokenizer...")
try:
    device = utils.get_device()
    tokenizer = Word2VecTokenizer()
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)  #

    # Load Query Tower from the checkpoint
    query_tower = QueryTower(
        tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(
        device
    )  #
    query_tower.load_state_dict(checkpoint["query_tower"])  #
    query_tower.eval()  # Set to evaluation mode

    # Load Doc Tower from the checkpoint
    doc_tower = DocTower(
        embeddings=tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(
        device
    )  #
    doc_tower.load_state_dict(checkpoint["doc_tower"])  #
    doc_tower.eval()  # Set to evaluation mode

    print("--> Models and tokenizer loaded successfully.")
except FileNotFoundError:
    print("\n[ERROR] Could not find the model file at 'data/models.pth'.")
    print("Please make sure you have a trained model saved at that location.")
    exit()

# --- 2. Define Test Sentences ---
# A list of diverse sentences. If the model has collapsed, the embeddings
# for all of these will be identical.
test_texts = [
    "does human hair stop squirrels",
    "what is the capital of the united kingdom",
    "Definition of the Smooth ER. The smooth endoplasmic reticulum...",
    "PyTorch is a popular deep learning framework.",
]
print(f"\n--> Testing with {len(test_texts)} different sentences.")


# --- 3. Helper Function for Encoding ---
def encode_text(text, tower, tokenizer, device):
    """Tokenizes and encodes a single string using the provided tower."""
    with torch.no_grad():
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(
            device
        )  #

        # Get the embedding from the tower
        embedding = tower(tokenized).cpu().numpy()
        return embedding


# --- 4. Verify the Query Tower ---
print("\n--- Verifying Query Tower ---")
query_embeddings = [
    encode_text(text, query_tower, tokenizer, device) for text in test_texts
]

# Check the L2 norm of each embedding. They should be identical if collapsed.
for i, text in enumerate(test_texts):
    norm = np.linalg.norm(query_embeddings[i])
    print(f"  Query: '{text[:30]}...' | Embedding Norm: {norm:.6f}")

# Programmatically check if the first two embeddings are the same
difference = np.sum(np.abs(query_embeddings[0] - query_embeddings[1]))
print(f"\n  Sum of absolute difference between first two embeddings: {difference:.8f}")

if difference < 1e-6:  # Using a small threshold for floating point comparison
    print(
        "  >> Verdict: The QueryTower has collapsed. It produces the same vector for different inputs."
    )
else:
    print("  >> Verdict: The QueryTower appears to be functioning correctly.")

# --- 5. Verify the Doc Tower ---
print("\n--- Verifying Doc Tower ---")
doc_embeddings = [
    encode_text(text, doc_tower, tokenizer, device) for text in test_texts
]

for i, text in enumerate(test_texts):
    norm = np.linalg.norm(doc_embeddings[i])
    print(f"  Doc Text: '{text[:30]}...' | Embedding Norm: {norm:.6f}")

difference_docs = np.sum(np.abs(doc_embeddings[0] - doc_embeddings[1]))
print(
    f"\n  Sum of absolute difference between first two embeddings: {difference_docs:.8f}"
)

if difference_docs < 1e-6:
    print(
        "  >> Verdict: The DocTower has collapsed. It produces the same vector for different inputs."
    )
else:
    print("  >> Verdict: The DocTower appears to be functioning correctly.")

print("\n--- Verification Script Finished ---")
