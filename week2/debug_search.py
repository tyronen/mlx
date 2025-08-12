# debug_search.py
import numpy as np
import redis
import torch
import utils
from model import QueryTower
from tokenizer import Word2VecTokenizer


def debug_search():
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

    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # Test with a simple query
    query = "human hair squirrels"

    with torch.no_grad():
        tokenized_query = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)

        query_embedding = query_tower(tokenized_query).cpu().numpy().astype(np.float32)
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding norm: {np.linalg.norm(query_embedding):.8f}")

        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        print(f"Normalized query embedding norm: {np.linalg.norm(query_embedding):.8f}")

        # Test with a few stored documents
        test_doc_ids = ["doc:train_76087_4", "doc:train_95792_3"]

        for doc_id in test_doc_ids:
            embedding_bytes = redis_client.hget(doc_id, "embedding")
            if embedding_bytes:
                doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                doc_embedding = doc_embedding / np.linalg.norm(
                    doc_embedding
                )  # Normalize

                # Calculate cosine similarity manually
                cosine_sim = np.dot(query_embedding.flatten(), doc_embedding)
                print(f"{doc_id}: cosine similarity = {cosine_sim:.6f}")

                # Get the text
                text = redis_client.hget(doc_id, "text").decode("utf-8")
                print(f"  Text: {text[:100]}...")
                print()


if __name__ == "__main__":
    debug_search()
