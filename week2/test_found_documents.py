# test_found_documents.py
import redis
import torch
import numpy as np
import utils
from model import QueryTower, DocTower
from tokenizer import Word2VecTokenizer


def test_found_documents():
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

    # Test the query
    query_text = "does human hair stop squirrels"

    with torch.no_grad():
        tokenized_query = tokenizer(
            query_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)

        query_embedding = query_tower(tokenized_query).cpu().numpy().astype(np.float32)
        query_embedding = query_embedding.flatten()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Test with the found documents
    found_docs = ["doc:train_77741_2", "doc:test_0_2", "doc:test_0_5"]

    print(f"Query: '{query_text}'")
    print("=" * 60)

    for doc_id in found_docs:
        # Get the stored embedding from Redis
        embedding_bytes = redis_client.hget(doc_id, "embedding")
        text_bytes = redis_client.hget(doc_id, "text")

        if embedding_bytes and text_bytes:
            # Get stored embedding
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)

            text = text_bytes.decode("utf-8")

            # Calculate similarity with stored embedding
            similarity = np.dot(query_embedding, stored_embedding)

            print(f"\nDocument: {doc_id}")
            print(f"Text: {text}")
            print(f"Similarity with stored embedding: {similarity:.6f}")
            print(f"Redis distance would be: {1 - similarity:.6f}")
            print("-" * 60)


if __name__ == "__main__":
    test_found_documents()
