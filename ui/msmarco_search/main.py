# type: ignore[assignment,attr-defined,index]
import json
import logging
import struct
import os
import numpy as np
import redis
import torch
from models import msmarco_search
from common import utils
from redis.commands.search.query import Query
import random
from typing import List, Dict, Any

from models.msmarco_tokenizer import Word2VecTokenizer

# Write the model version here or find some way to derive it from the model
# eg. from the model files name
model_version = "0.1.0"


utils.setup_logging()
device = utils.get_device()
tokenizer = Word2VecTokenizer()
checkpoint = torch.load(msmarco_search.MODEL_FILE, map_location=device)

query_tower = msmarco_search.QueryTower(
    tokenizer.embeddings,
    embed_dim=checkpoint["embed_dim"],
    dropout_rate=checkpoint["dropout_rate"],
).to(device)
query_tower.load_state_dict(checkpoint["query_tower"])
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_client: redis.Redis = redis.Redis(host=redis_host, port=redis_port, db=0)


# Define the endpoints
def ping():
    return "ok"


def version():
    return {"version": model_version}


def search(query):
    global query_tower, redis_client, tokenizer, device
    if (query_tower is None) or (tokenizer is None) or (redis_client is None):
        raise Exception("App not initialized.")

    start_time = os.times().elapsed

    # Get search results
    documents = do_search(query, query_tower, redis_client, tokenizer, device)

    # Get ground truth
    ground_truth_docs = get_ground_truth_docs(query)
    ground_truth_similarities = calculate_ground_truth_similarity(
        query, ground_truth_docs
    )

    end_time = os.times().elapsed
    latency = (end_time - start_time) * 1000

    # Check if top result matches ground truth
    has_match = False
    if documents and ground_truth_docs:
        top_doc_id = documents[0][0]
        gt_doc_ids = [gt["doc_id"] for gt in ground_truth_docs]
        has_match = any(gt_id in top_doc_id for gt_id in gt_doc_ids)

        logging.info(f"Query: {query}")
        logging.info(f"Top result: {documents[0]}")
        logging.info(f"Ground truth: {ground_truth_docs[0]}")

    response = {
        "documents": documents,
        "ground_truth": ground_truth_similarities,
        "query": query,
        "latency": int(latency),
        "version": model_version,
        "evaluation": {
            "has_ground_truth": len(ground_truth_docs) > 0,
            "top_result_matches_gt": has_match,
            "num_ground_truth_docs": len(ground_truth_docs),
        },
    }

    message = {
        "Latency": int(latency),
        "Version": model_version,
        "Input": query,
        "Document": documents[0][:50] if documents else None,
        "HasGroundTruth": len(ground_truth_docs) > 0,
        "MatchesGroundTruth": has_match,
    }

    return response


def do_search(query, query_tower, redis_client, tokenizer, device, top_k=5):
    query_tower.eval()
    with torch.no_grad():
        tokenized_query = tokenizer(query)["input_ids"].to(device)

        query_embedding = (
            query_tower(tokenized_query).cpu().numpy().astype(np.float32).flatten()
        )

        # Convert to bytes for Redis
        query_embedding_bytes = struct.pack(
            f"<{len(query_embedding)}f", *query_embedding
        )

    redis_query = f"*=>[KNN {top_k} @embedding $vec_param AS vector_score]"
    query_obj = (
        Query(redis_query)
        .return_fields("text", "vector_score")
        .sort_by("vector_score")
        .paging(0, top_k)
        .dialect(2)
    )

    results = redis_client.ft("doc_index").search(
        query_obj, query_params={"vec_param": query_embedding_bytes}
    )

    # Return results with proper similarity scores
    search_results = []
    for doc in results.docs:
        doc_id = doc.id
        distance = float(doc.vector_score)
        similarity = 1.0 - distance  # Convert distance to similarity
        text = doc.text
        search_results.append((doc_id, similarity, text))

    return search_results


def get_ground_truth_docs(query: str) -> List[Dict[str, Any]]:
    """Get ground truth positive documents for a query"""
    global redis_client
    try:
        positive_doc_ids = redis_client.smembers(f"query_positive:{query}")
        positive_docs: List[Dict[str, Any]] = []

        for doc_id in positive_doc_ids:
            doc_id_str = (
                "doc:" + doc_id.decode("utf-8") if isinstance(doc_id, bytes) else doc_id
            )
            doc_data = redis_client.hmget(f"{doc_id_str}", ["text"])
            if doc_data and doc_data[0]:
                text = (
                    doc_data[0].decode("utf-8")
                    if isinstance(
                        doc_data[0], bytes
                    )  # pyright: ignore[reportIndexIssue]
                    else doc_data[0]
                )
                positive_docs.append(
                    {
                        "doc_id": doc_id_str,
                        "text": text,
                        "similarity": 1.0,  # Ground truth
                    }
                )

        return positive_docs
    except Exception as e:
        logging.error(f"Error getting ground truth for query '{query}': {e}")
        return []


def calculate_ground_truth_similarity(
    query: str, ground_truth_docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    global query_tower, tokenizer, device
    """Calculate actual similarity between query and ground truth docs"""
    if not ground_truth_docs:
        return []

    query_tower.eval()
    with torch.no_grad():
        # Get query embedding
        tokenized_query = tokenizer(query)["input_ids"].to(device)
        query_embedding = query_tower(tokenized_query).cpu().numpy().flatten()

        # Calculate similarities to ground truth docs
        results: List[Dict[str, Any]] = []
        for gt_doc in ground_truth_docs:
            doc_id = gt_doc["doc_id"]
            # Get document embedding from Redis
            doc_embedding_bytes = redis_client.hget(f"{doc_id}", "embedding")
            if doc_embedding_bytes:
                # Ensure we have bytes data
                if isinstance(doc_embedding_bytes, str):
                    doc_embedding_bytes = doc_embedding_bytes.encode("utf-8")
                doc_embedding = np.frombuffer(doc_embedding_bytes, dtype=np.float32)  # type: ignore[arg-type]
                similarity = np.dot(query_embedding, doc_embedding)
                results.append(
                    {
                        "doc_id": doc_id,
                        "text": gt_doc["text"],
                        "similarity": float(similarity),
                    }
                )

        return results


def get_random_query() -> Dict[str, Any]:
    """Get a random query that has ground truth data"""
    all_queries = redis_client.smembers("all_queries")
    if not all_queries:
        return {"error": "No queries found in database"}

    # Convert bytes to strings and pick random
    query_list = [q.decode("utf-8") if isinstance(q, bytes) else q for q in all_queries]
    random_query = random.choice(query_list)

    return search(random_query)
