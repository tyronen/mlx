# find_specific_doc.py
import redis


def find_hair_squirrel_doc():
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # Search for documents containing key terms
    search_terms = ["human hair", "squirrels", "vegetable", "flower gardens", "scare"]

    found_docs = []
    doc_keys = redis_client.keys("doc:*")

    print(f"Searching through {len(doc_keys)} documents...")

    for i, key in enumerate(doc_keys):
        if i % 10000 == 0:
            print(f"Processed {i} documents...")

        key_str = key.decode("utf-8")
        text_bytes = redis_client.hget(key, "text")

        if text_bytes:
            text = text_bytes.decode("utf-8").lower()

            # Look for the specific content
            if "human hair" in text and "squirrel" in text:
                found_docs.append((key_str, text))
                print(f"\n🎯 FOUND RELEVANT DOC: {key_str}")
                print(f"Text: {text}")

                # Get the embedding for this document
                embedding_bytes = redis_client.hget(key, "embedding")
                if embedding_bytes:
                    import numpy as np

                    doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    print(f"Embedding norm: {np.linalg.norm(doc_embedding):.8f}")
                print("-" * 80)

    if not found_docs:
        print("❌ No documents found containing 'human hair' and 'squirrel'")

        # Try broader search
        hair_docs = []
        squirrel_docs = []

        for key in doc_keys[:1000]:  # Check first 1000
            text_bytes = redis_client.hget(key, "text")
            if text_bytes:
                text = text_bytes.decode("utf-8").lower()
                if "hair" in text:
                    hair_docs.append(key.decode("utf-8"))
                if "squirrel" in text:
                    squirrel_docs.append(key.decode("utf-8"))

        print(f"Found {len(hair_docs)} docs with 'hair'")
        print(f"Found {len(squirrel_docs)} docs with 'squirrel'")

    return found_docs


if __name__ == "__main__":
    find_hair_squirrel_doc()
