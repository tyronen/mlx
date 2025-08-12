# test_anchor_adapter.py
import torch
import numpy as np
from model import QueryTower, DocTower
from tokenizer import Word2VecTokenizer
import utils


def test_anchor_adapter():
    device = utils.get_device()
    tokenizer = Word2VecTokenizer()

    # Test with FROZEN query embeddings (like doc tower)
    query_tower_frozen = QueryTower(
        tokenizer.embeddings,
        embed_dim=300,
        dropout_rate=0.1,
    ).to(device)

    # Manually freeze the query embeddings
    query_tower_frozen.embedding.weight.requires_grad = False

    # Load your trained model (with adapted query embeddings)
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)
    query_tower_trained = QueryTower(
        tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    query_tower_trained.load_state_dict(checkpoint["query_tower"])

    doc_tower = DocTower(
        tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    doc_tower.load_state_dict(checkpoint["doc_tower"])

    # Test texts
    query_text = "does human hair stop squirrels"
    answer_text = "Spread some human hair around your vegetable and flower gardens. This will scare the squirrels away because humans are predators of squirrels."

    with torch.no_grad():
        # Query with FROZEN embeddings (original Word2Vec space)
        query_tokens = tokenizer(
            query_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)
        query_frozen = query_tower_frozen(query_tokens).cpu().numpy().flatten()
        query_frozen = query_frozen / np.linalg.norm(query_frozen)

        # Query with TRAINED embeddings (adapted space)
        query_trained = query_tower_trained(query_tokens).cpu().numpy().flatten()
        query_trained = query_trained / np.linalg.norm(query_trained)

        # Document (always frozen)
        doc_tokens = tokenizer(
            answer_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )["input_ids"].to(device)
        doc_emb = doc_tower(doc_tokens).cpu().numpy().flatten()
        doc_emb = doc_emb / np.linalg.norm(doc_emb)

        # Compare similarities
        sim_frozen = np.dot(query_frozen, doc_emb)
        sim_trained = np.dot(query_trained, doc_emb)

        print(f"Query: '{query_text}'")
        print(f"Answer: '{answer_text[:80]}...'")
        print(f"\n📊 SIMILARITY COMPARISON:")
        print(f"Frozen query embeddings:  {sim_frozen:.6f}")
        print(f"Trained query embeddings: {sim_trained:.6f}")
        print(
            f"\nTraining {'✅ improved' if sim_trained > sim_frozen else '❌ hurt'} similarity"
        )

        # Check how much the query embedding changed
        query_change = np.linalg.norm(query_trained - query_frozen)
        print(f"\nQuery embedding change magnitude: {query_change:.6f}")


if __name__ == "__main__":
    test_anchor_adapter()
