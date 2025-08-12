# analyze_redis_duplicates.py
import redis
import numpy as np
from collections import defaultdict


def analyze_duplicates():
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # Get all document keys
    doc_keys = redis_client.keys("doc:*")
    print(f"Total documents in Redis: {len(doc_keys)}")

    text_to_docs = defaultdict(list)
    embedding_similarities = []

    # Group documents by text content
    for key in doc_keys[:100]:  # Check first 100 docs
        key_str = key.decode("utf-8")
        text = redis_client.hget(key, "text")
        if text:
            text_str = text.decode("utf-8")
            text_to_docs[text_str].append(key_str)

    # Find duplicates
    duplicates = {text: docs for text, docs in text_to_docs.items() if len(docs) > 1}
    print(f"\nFound {len(duplicates)} duplicate text entries:")

    for text, doc_ids in list(duplicates.items())[:5]:  # Show first 5
        print(f"\nText: '{text[:50]}...'")
        print(f"Document IDs: {doc_ids}")

        # Check if embeddings are also identical
        embeddings = []
        for doc_id in doc_ids:
            emb_bytes = redis_client.hget(doc_id, "embedding")
            if emb_bytes:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                embeddings.append(emb)

        if len(embeddings) > 1:
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            print(f"Embedding similarity: {similarity:.8f}")


if __name__ == "__main__":
    analyze_duplicates()
