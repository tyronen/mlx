# check_redis_data.py
#
# A script to fetch and inspect the embedding vectors as they are
# stored in the Redis database.

import redis
import numpy as np
import torch
import utils
from model import QueryTower
from tokenizer import Word2VecTokenizer

print("--- Redis Data Verification Script ---")

# --- 1. Generate a 'correct' query vector using the verified model ---
print("--> Generating a known-good query vector...")
try:
    device = utils.get_device()
    tokenizer = Word2VecTokenizer()
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)  #
    query_tower = QueryTower(
        tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(
        device
    )  #
    query_tower.load_state_dict(checkpoint["query_tower"])  #
    query_tower.eval()

    query_text = "does human hair stop squirrels"
    with torch.no_grad():
        tokenized_query = tokenizer(
            query_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(
            device
        )  #
        correct_query_embedding = query_tower(tokenized_query).cpu().numpy()

    print(f"--> Generated correct embedding for '{query_text}'.")
except FileNotFoundError:
    print("\n[ERROR] Could not find the model file at 'data/models.pth'.")
    exit()

# --- 2. Connect to Redis and fetch stored embeddings ---
print("\n--> Fetching stored vectors from Redis...")
try:
    # NOTE: Your main.py connects to 'redis-stack', but your store_documents.py
    # connects to 'localhost'. Make sure this matches how you run your services.
    # If using Docker Compose, 'redis-stack' is likely correct.
    # If running locally, 'localhost' is correct.
    redis_client = redis.Redis(host="localhost", port=6379, db=0)  #
    redis_client.ping()
    print("--> Redis connection successful.")
except redis.exceptions.ConnectionError as e:
    print(f"\n[ERROR] Could not connect to Redis. Is it running? Error: {e}")
    exit()

# Use the document IDs from your example output
doc_ids_to_check = [
    "doc:train_76087_4",
    "doc:train_95792_3",
    "doc:train_85584_1",
]
stored_embeddings = {}

for doc_id in doc_ids_to_check:
    try:
        # Fetch the raw bytes for the 'embedding' field from the HASH
        embedding_bytes = redis_client.hget(doc_id, "embedding")  #
        if embedding_bytes:
            # Convert the byte string back into a numpy array
            stored_vector = np.frombuffer(embedding_bytes, dtype=np.float32)
            stored_embeddings[doc_id] = stored_vector
            print(
                f"  Fetched vector for {doc_id}. Norm: {np.linalg.norm(stored_vector):.6f}"
            )
        else:
            print(f"  [WARN] Could not find document with ID: {doc_id}")
    except Exception as e:
        print(f"  [ERROR] Failed to fetch or process {doc_id}. Error: {e}")

# --- 3. Compare the vectors fetched from Redis ---
print("\n--- Analysis of Stored Vectors ---")
if len(stored_embeddings) > 1:
    # Compare the first two stored embeddings to see if they are identical
    ids = list(stored_embeddings.keys())
    vec1 = stored_embeddings[ids[0]]
    vec2 = stored_embeddings[ids[1]]

    difference = np.sum(np.abs(vec1 - vec2))
    print(
        f"--> Sum of absolute difference between '{ids[0]}' and '{ids[1]}': {difference:.8f}"
    )
    if difference < 1e-6:
        print(
            "  >> Verdict: The vectors for different documents in Redis ARE IDENTICAL."
        )
        print("  >> This confirms the problem is likely in 'store_documents.py'.")
    else:
        print(
            "  >> Verdict: The vectors in Redis are different. The storage script seems okay."
        )

else:
    print("  Not enough embeddings were fetched to perform a comparison.")
