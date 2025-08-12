# test_correct_similarity.py
import redis
import torch
import numpy as np
import utils
from model import QueryTower
from tokenizer import Word2VecTokenizer


def test_specific_similarity():
    # Load model
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

    # Generate query embedding
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

    # Test with the expected text
    expected_text = "Spread some human hair around your vegetable and flower gardens. This will scare the squirrels away because humans are predators of squirrels. It is better if the hair hasn't been washed so the squirrels will easily pick up the human scent."

    # Tokenize and encode the expected answer
    with torch.no_grad():
        # Load document tower
        doc_tower = DocTower(
            embeddings=tokenizer.embeddings,
            embed_dim=checkpoint["embed_dim"],
            dropout_rate=checkpoint["dropout_rate"],
        ).to(device)
        doc_tower.load_state_dict(checkpoint["doc_tower"])
        doc_tower.eval()

        tokenized_doc = tokenizer(
            expected_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)

        expected_doc_embedding = (
            doc_tower(tokenized_doc).cpu().numpy().astype(np.float32)
        )
        expected_doc_embedding = expected_doc_embedding.flatten()
        expected_doc_embedding = expected_doc_embedding / np.linalg.norm(
            expected_doc_embedding
        )

    # Calculate similarity
    similarity = np.dot(query_embedding, expected_doc_embedding)
    print(f"Query: '{query_text}'")
    print(f"Expected answer: '{expected_text[:100]}...'")
    print(f"Similarity between query and expected answer: {similarity:.6f}")

    # Compare with the documents we're actually getting
    current_results = [
        "A limit of razor clams dug in the wet sand at Agate Beach. The razor clams in the photo on the right were dug from the Cove in Seaside Oregon. The book includes:Oregon's Razor Clams discloses all of",
        "Average Senior Fitness Instructor Salaries. The average salary for senior fitness instructor jobs is $47,000. Average senior fitness instructor salaries can vary greatly due to company, location",
    ]

    print("\nComparing with current top results:")
    for i, bad_text in enumerate(current_results):
        with torch.no_grad():
            tokenized_bad = tokenizer(
                bad_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            )["input_ids"].to(device)

            bad_doc_embedding = (
                doc_tower(tokenized_bad).cpu().numpy().astype(np.float32)
            )
            bad_doc_embedding = bad_doc_embedding.flatten()
            bad_doc_embedding = bad_doc_embedding / np.linalg.norm(bad_doc_embedding)

        bad_similarity = np.dot(query_embedding, bad_doc_embedding)
        print(f"  {i + 1}. Similarity with '{bad_text[:50]}...': {bad_similarity:.6f}")


if __name__ == "__main__":
    # Import the missing DocTower
    from model import DocTower

    test_specific_similarity()
