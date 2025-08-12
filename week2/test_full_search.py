# test_full_search.py
import redis
import torch
import numpy as np
from redis.commands.search.query import Query
import utils
from model import QueryTower
from tokenizer import Word2VecTokenizer


def test_search_pipeline():
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

    # Test query
    query_text = "does human hair stop squirrels"

    print(f"Testing search for: '{query_text}'")
    print("=" * 50)

    # Generate query embedding
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

    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding norm: {np.linalg.norm(query_embedding):.8f}")

    # Test different Redis query formats
    test_queries = [
        f"*=>[KNN 5 @embedding $vec AS score]",
        f"*=>[KNN 5 @embedding $vec_param AS vector_score]",
        f"*=>[KNN 5 @embedding $vec_param]",
    ]

    for i, redis_query in enumerate(test_queries):
        print(f"\n--- Test {i + 1}: {redis_query} ---")

        try:
            query_obj = (
                Query(redis_query)
                .return_fields(
                    "text",
                    (
                        "score"
                        if "AS score" in redis_query
                        else "vector_score" if "AS vector_score" in redis_query else ""
                    ),
                )
                .sort_by(
                    "score"
                    if "AS score" in redis_query
                    else (
                        "vector_score"
                        if "AS vector_score" in redis_query
                        else "__score"
                    )
                )
                .paging(0, 5)
                .dialect(2)
            )

            results = redis_client.ft("doc_index").search(
                query_obj,
                query_params={
                    "vec": query_embedding.tobytes(),
                    "vec_param": query_embedding.tobytes(),
                },
            )

            print(f"Found {len(results.docs)} results:")
            for j, doc in enumerate(results.docs):
                # Get all available attributes
                attrs = vars(doc)
                print(f"  {j + 1}. {doc.id}")
                print(f"     Attributes: {list(attrs.keys())}")
                if hasattr(doc, "text"):
                    print(f"     Text: {doc.text[:80]}...")
                if hasattr(doc, "score"):
                    print(f"     Score: {doc.score}")
                if hasattr(doc, "vector_score"):
                    print(f"     Vector Score: {doc.vector_score}")
                print()

        except Exception as e:
            print(f"Error with query format: {e}")

    # Test manual similarity calculation
    print("\n--- Manual Similarity Check ---")
    sample_docs = redis_client.keys("doc:*")[:10]
    similarities = []

    for doc_key in sample_docs:
        doc_key_str = doc_key.decode("utf-8")
        embedding_bytes = redis_client.hget(doc_key, "embedding")
        text_bytes = redis_client.hget(doc_key, "text")

        if embedding_bytes and text_bytes:
            doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)
            text = text_bytes.decode("utf-8")

            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((doc_key_str, similarity, text))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 most similar documents (manual calculation):")
    for i, (doc_id, sim, text) in enumerate(similarities[:5]):
        print(f"  {i + 1}. {doc_id} (similarity: {sim:.6f})")
        print(f"     Text: {text[:80]}...")
        print()


if __name__ == "__main__":
    test_search_pipeline()
