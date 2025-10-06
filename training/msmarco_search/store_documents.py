import logging
from pathlib import Path
import shutil
import time

import redis
import numpy as np
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F

from common import utils
from models import msmarco_search, msmarco_tokenizer


def load_document_corpus():
    """Load all unique documents from MS MARCO dataset"""
    logging.info("Loading MS MARCO document corpus...")

    # Load the dataset
    ms_marco_data = load_dataset("ms_marco", "v1.1")

    # Extract all unique documents from all splits
    all_documents = {}  # Use dict to avoid duplicates
    query_to_positive_docs = {}  # Map query -> list of positive doc IDs
    query_to_all_docs = {}  # Map query -> list of all doc IDs

    for split_name in ["train", "validation", "test"]:
        logging.info(f"Processing {split_name} split...")
        split_data = ms_marco_data[split_name]

        for row in tqdm(split_data, desc=f"Extracting docs from {split_name}"):
            query = row["query"]
            query_id = row.get("query_id", f"{split_name}_{len(query_to_all_docs)}")
            passages = row["passages"]
            passage_texts = passages["passage_text"]
            is_selected = passages.get("is_selected", [False] * len(passage_texts))

            # Initialize query tracking
            if query not in query_to_positive_docs:
                query_to_positive_docs[query] = []
                query_to_all_docs[query] = []

            # Each passage gets a unique ID
            for i, passage_text in enumerate(passage_texts):
                # Create unique document ID
                doc_id = f"{split_name}_{query_id}_{i}"

                # Store document
                all_documents[doc_id] = {
                    "id": doc_id,
                    "text": passage_text,
                    "split": split_name,
                    "query_id": query_id,
                    "is_selected": is_selected[i] if i < len(is_selected) else False,
                }

                # Track query-document relationships
                query_to_all_docs[query].append(doc_id)
                if is_selected[i] if i < len(is_selected) else False:
                    query_to_positive_docs[query].append(doc_id)

    # Convert to list
    documents = list(all_documents.values())
    logging.info(f"Loaded {len(documents)} unique documents from MS MARCO")
    logging.info(
        f"Tracked {len(query_to_positive_docs)} positives, {len(query_to_all_docs)} total"
    )

    return documents, query_to_positive_docs, query_to_all_docs


def encode_all_documents(tokenizer, doc_tower, documents, device, batch_size=1000):
    """Encode all documents and return embeddings with IDs"""
    doc_tower.eval()

    all_embeddings = []
    doc_metadata = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(documents) // 20, batch_size), desc="Encoding documents"
        ):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [doc["text"] for doc in batch_docs]

            # Tokenize batch
            tokenized = tokenizer(batch_texts)["input_ids"].to(device)

            # Encode and normalize
            embeddings = doc_tower(tokenized)
            embeddings = F.normalize(
                embeddings, p=2, dim=1
            )  # Normalize like in model.py

            all_embeddings.extend(embeddings.cpu().numpy())
            # Store metadata for each document
            for doc in batch_docs:
                doc_metadata.append(
                    {
                        "id": doc["id"],
                        "text": (
                            doc["text"][:200] + "..."
                            if len(doc["text"]) > 200
                            else doc["text"]
                        ),
                        # Truncate for storage
                        "split": doc["split"],
                        "query_id": doc.get("query_id"),
                        "is_selected": doc.get("is_selected", False),
                    }
                )
    logging.info(
        f"Saving {len(doc_metadata)} metadata and {len(all_embeddings)} embeddings to Redis"
    )
    return doc_metadata, np.array(all_embeddings)


def store_embeddings_in_redis(doc_metadata, embeddings, redis_client):
    """Store document embeddings in Redis"""

    pipeline = redis_client.pipeline(transaction=False)
    for doc_meta, embedding in tqdm(
        zip(doc_metadata, embeddings), desc="Storing in Redis"
    ):
        # Store embedding as binary data
        embedding_bytes = embedding.astype(np.float32).tobytes()
        doc_id = doc_meta["id"]
        pipeline.hset(
            f"doc:{doc_id}",
            mapping={"text": doc_meta["text"], "embedding": embedding_bytes},
        )

        if len(pipeline.command_stack) >= 1000:
            pipeline.execute()
    pipeline.execute()

    logging.info(f"Stored {len(doc_metadata)} document embeddings in Redis")


def store_query_mappings_in_redis(
    query_to_positive_docs, query_to_all_docs, redis_client
):
    """Store query-to-document mappings in Redis"""
    logging.info("Storing query mappings in Redis...")

    pipeline = redis_client.pipeline(transaction=False)

    # Store positive document mappings
    for query, doc_ids in tqdm(query_to_positive_docs.items(), "Query to positive"):
        if doc_ids:  # skip empty sets
            pipeline.sadd(f"query_positive:{query}", *doc_ids)

    # Store all document mappings
    for query, doc_ids in tqdm(query_to_all_docs.items(), "Query to docs"):
        if doc_ids:
            pipeline.sadd(f"query_all:{query}", *doc_ids)

    # Store a set of all queries for random selection
    all_queries = list(query_to_positive_docs.keys())
    pipeline.sadd("all_queries", *all_queries)

    pipeline.execute()
    logging.info(f"Stored mappings for {len(query_to_positive_docs)} queries")


def create_redis_index(redis_client, dim):
    try:
        redis_client.ft("doc_index").dropindex(delete_documents=False)
    except Exception:
        pass

    logging.info("Creating Redis index...")
    redis_client.ft("doc_index").create_index(
        fields=[
            TextField("text"),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ],
        definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH),
    )


def create_redis_dump(redis_client):
    """Create a Redis dump file for production"""
    logging.info("💾 Creating Redis dump...")
    try:
        redis_client.save()

    except Exception as e:
        logging.error(f"❌ Failed to create Redis dump: {e}")
        return None


def main():
    utils.setup_logging()
    device = utils.get_device()
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    redis_client.flushdb()
    checkpoint = torch.load(msmarco_search.MODEL_FILE, map_location=device)
    tokenizer = msmarco_tokenizer.Word2VecTokenizer()
    doc_tower = msmarco_search.DocTower(
        embeddings=tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    doc_tower.load_state_dict(checkpoint["doc_tower"])

    logging.info("Loading MS MARCO dataset...")
    documents, query_to_positive_docs, query_to_all_docs = load_document_corpus()

    store_query_mappings_in_redis(
        query_to_positive_docs, query_to_all_docs, redis_client
    )

    logging.info("Encoding documents...")
    doc_metadata, embeddings = encode_all_documents(
        tokenizer, doc_tower, documents, device
    )

    logging.info("Storing in Redis...")
    store_embeddings_in_redis(doc_metadata, embeddings, redis_client)

    create_redis_index(redis_client, embeddings.shape[1])

    # Store metadata
    redis_client.set("embedding_dim", embeddings.shape[1])
    redis_client.set("total_docs", len(doc_metadata))
    redis_client.set("total_queries", len(query_to_positive_docs))
    logging.info("Document cache setup complete!")
    create_redis_dump(redis_client)


if __name__ == "__main__":
    main()
