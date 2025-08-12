import json
import logging
import struct
from contextlib import asynccontextmanager
import os
import numpy as np
import redis
import torch
from fastapi import FastAPI
from model import QueryTower
import utils
from redis.commands.search.query import Query
import random

from tokenizer import Word2VecTokenizer

# Write the model version here or find some way to derive it from the model
# eg. from the model files name
model_version = "0.1.0"

# Set the log path.
# This should be a directory that is writable by the application.
# In a docker container, you can use /var/log/ as the directory.
# Mount this directory to a volume on the host machine to persist the logs.
log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"


query_tower = None
tokenizer = None
device = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_tower, tokenizer, device, redis_client
    utils.setup_logging()
    device = utils.get_device()
    tokenizer = Word2VecTokenizer()
    checkpoint = torch.load(utils.MODEL_FILE, map_location=device)

    query_tower = QueryTower(
        tokenizer.embeddings,
        embed_dim=checkpoint["embed_dim"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(device)
    query_tower.load_state_dict(checkpoint["query_tower"])
    redis_client = redis.Redis(host="redis-stack", port=6379, db=0)
    yield


app = FastAPI(lifespan=lifespan)


# Define the endpoints
@app.get("/ping")
def ping():
    return "ok"


@app.get("/version")
def version():
    return {"version": model_version}


@app.get("/logs")
def logs():
    return read_logs(log_path)


@app.get("/search")
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

    log_request(log_path, json.dumps(message))
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

        # --- START DEBUG CODE ---
        # Save the numpy array to a file for external verification
        np.save("debug_query_vector.npy", query_embedding)
        # --- END DEBUG CODE ---

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


def get_ground_truth_docs(query):
    """Get ground truth positive documents for a query"""
    global redis_client
    try:
        positive_doc_ids = redis_client.smembers(f"query_positive:{query}")
        positive_docs = []

        for doc_id in positive_doc_ids:
            doc_id_str = (
                "doc:" + doc_id.decode("utf-8") if isinstance(doc_id, bytes) else doc_id
            )
            doc_data = redis_client.hmget(f"{doc_id_str}", "text")
            if doc_data[0]:
                text = (
                    doc_data[0].decode("utf-8")
                    if isinstance(doc_data[0], bytes)
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


def calculate_ground_truth_similarity(query, ground_truth_docs):
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
        results = []
        for gt_doc in ground_truth_docs:
            doc_id = gt_doc["doc_id"]
            # Get document embedding from Redis
            doc_embedding_bytes = redis_client.hget(f"{doc_id}", "embedding")
            if doc_embedding_bytes:
                doc_embedding = np.frombuffer(doc_embedding_bytes, dtype=np.float32)
                similarity = np.dot(query_embedding, doc_embedding)
                results.append(
                    {
                        "doc_id": doc_id,
                        "text": gt_doc["text"],
                        "similarity": float(similarity),
                    }
                )

        return results


@app.get("/random_query")
def get_random_query():
    """Get a random query that has ground truth data"""
    all_queries = redis_client.smembers("all_queries")
    if not all_queries:
        return {"error": "No queries found in database"}

    # Convert bytes to strings and pick random
    query_list = [q.decode("utf-8") if isinstance(q, bytes) else q for q in all_queries]
    random_query = random.choice(query_list)

    return search(random_query)


##### Log The Request #####
def log_request(log_path, message):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")


##### Read The Logs #####
def read_logs(log_path):
    # read the logs from the log_path
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as log_file:
        lines = log_file.readlines()
    return [line.strip() for line in lines]
