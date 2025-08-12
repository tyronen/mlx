import numpy as np
import torch
import utils
from model import QueryTower
from tokenizer import Word2VecTokenizer

# --- 1. Generate the 'golden' vector using the same logic as your working test script ---
print("--- Generating golden vector ---")
device = utils.get_device()
tokenizer = Word2VecTokenizer()
checkpoint = torch.load(utils.MODEL_FILE, map_location=device)
query_tower = QueryTower(
    tokenizer.embeddings,
    embed_dim=checkpoint["embed_dim"],
    dropout_rate=checkpoint["dropout_rate"],
).to(device)
query_tower.load_state_dict(checkpoint["query_tower"])
query_tower.eval()
query_text = "does human hair stop squirrels"
with torch.no_grad():
    tokenized = tokenizer(
        query_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )["input_ids"].to(device)
    golden_vector = query_tower(tokenized).cpu().numpy()
    golden_vector = golden_vector / np.linalg.norm(golden_vector)
print("Golden vector generated.")

# --- 2. Load the debug vector that was saved by main.py ---
print("\n--- Loading debug vector from main.py ---")
try:
    debug_vector = np.load("debug_query_vector.npy")
    print("Debug vector loaded.")
except FileNotFoundError:
    print("[ERROR] debug_query_vector.npy not found!")
    print("Please run a search on the FastAPI endpoint first to generate the file.")
    exit()

# --- 3. Compare the two vectors ---
print("\n--- Comparing vectors ---")
difference = np.sum(np.abs(golden_vector - debug_vector))
print(f"Sum of absolute difference: {difference:.8f}")

if difference < 1e-6:
    print("\n>> Verdict: The vectors are IDENTICAL.")
    print(
        ">> This means vector generation is correct, and the issue must be in the Redis query itself."
    )
else:
    print("\n>> Verdict: The vectors are DIFFERENT.")
    print(">> This means the QueryTower or Tokenizer in main.py is in a corrupt state.")
