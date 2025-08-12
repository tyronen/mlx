import requests
import json
import time
from tqdm import tqdm
import numpy as np
import argparse
import logging

# Configuration
FASTAPI_URL = "http://localhost:8000"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_random_query():
    """Get a random query from the API"""
    try:
        response = requests.get(f"{FASTAPI_URL}/random_query", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def calculate_metrics(results_list):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        "total_queries": len(results_list),
        "queries_with_ground_truth": 0,
        "exact_matches": 0,
        "mrr_scores": [],
        "recall_at_1": 0,
        "recall_at_3": 0,
        "recall_at_5": 0,
        "average_latency": 0,
        "similarity_scores": [],
        "ground_truth_similarities": [],
        "failed_queries": 0,
    }

    total_latency = 0

    for result in results_list:
        if result is None:
            metrics["failed_queries"] += 1
            continue

        # Basic metrics
        total_latency += result.get("latency", 0)

        evaluation = result.get("evaluation", {})
        if not evaluation.get("has_ground_truth"):
            continue

        metrics["queries_with_ground_truth"] += 1

        # Exact match
        if evaluation.get("top_result_matches_gt"):
            metrics["exact_matches"] += 1
            metrics["recall_at_1"] += 1

        # Similarity scores
        documents = result.get("documents", [])
        if documents:
            top_similarity = documents[0][1]
            metrics["similarity_scores"].append(top_similarity)

        # Ground truth similarities
        ground_truth = result.get("ground_truth", [])
        if ground_truth:
            gt_similarities = [gt["similarity"] for gt in ground_truth]
            metrics["ground_truth_similarities"].extend(gt_similarities)

        # Calculate MRR and Recall@k
        mrr_score, recall_scores = calculate_ranking_metrics(result)
        if mrr_score is not None:
            metrics["mrr_scores"].append(mrr_score)
            if recall_scores["recall_at_3"]:
                metrics["recall_at_3"] += 1
            if recall_scores["recall_at_5"]:
                metrics["recall_at_5"] += 1

    # Calculate averages and percentages
    if metrics["queries_with_ground_truth"] > 0:
        metrics["exact_match_rate"] = (
            metrics["exact_matches"] / metrics["queries_with_ground_truth"]
        )
        metrics["recall_at_1_rate"] = (
            metrics["recall_at_1"] / metrics["queries_with_ground_truth"]
        )
        metrics["recall_at_3_rate"] = (
            metrics["recall_at_3"] / metrics["queries_with_ground_truth"]
        )
        metrics["recall_at_5_rate"] = (
            metrics["recall_at_5"] / metrics["queries_with_ground_truth"]
        )

    if metrics["mrr_scores"]:
        metrics["mean_mrr"] = np.mean(metrics["mrr_scores"])

    if metrics["total_queries"] > 0:
        metrics["average_latency"] = total_latency / metrics["total_queries"]

    if metrics["similarity_scores"]:
        metrics["mean_similarity"] = np.mean(metrics["similarity_scores"])
        metrics["std_similarity"] = np.std(metrics["similarity_scores"])

    if metrics["ground_truth_similarities"]:
        metrics["mean_gt_similarity"] = np.mean(metrics["ground_truth_similarities"])
        metrics["std_gt_similarity"] = np.std(metrics["ground_truth_similarities"])

    return metrics


def calculate_ranking_metrics(result):
    """Calculate MRR and Recall@k for a single query"""
    documents = result.get("documents", [])
    ground_truth = result.get("ground_truth", [])

    if not documents or not ground_truth:
        return None, {"recall_at_3": False, "recall_at_5": False}

    # Get ground truth document IDs
    gt_doc_ids = set()
    for gt in ground_truth:
        gt_doc_ids.add(gt["doc_id"])

    # Find rank of first relevant document
    mrr_score = 0
    recall_at_3 = False
    recall_at_5 = False

    for rank, (doc_id, similarity, text) in enumerate(documents, 1):
        # Check if this document matches any ground truth
        is_relevant = any(gt_id in doc_id for gt_id in gt_doc_ids)

        if is_relevant:
            if mrr_score == 0:  # First relevant document
                mrr_score = 1.0 / rank
            if rank <= 3:
                recall_at_3 = True
            if rank <= 5:
                recall_at_5 = True

    return mrr_score, {"recall_at_3": recall_at_3, "recall_at_5": recall_at_5}


def print_metrics(metrics):
    """Print evaluation metrics in a nice format"""
    print("\n" + "=" * 60)
    print("🔍 MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"📊 Dataset Coverage:")
    print(f"   Total queries tested: {metrics['total_queries']}")
    print(f"   Queries with ground truth: {metrics['queries_with_ground_truth']}")
    print(f"   Failed queries: {metrics['failed_queries']}")

    if metrics["queries_with_ground_truth"] > 0:
        print(f"\n🎯 Retrieval Performance:")
        print(
            f"   Exact Match Rate: {metrics.get('exact_match_rate', 0):.3f} ({metrics['exact_matches']}/{metrics['queries_with_ground_truth']})"
        )
        print(f"   Recall@1: {metrics.get('recall_at_1_rate', 0):.3f}")
        print(f"   Recall@3: {metrics.get('recall_at_3_rate', 0):.3f}")
        print(f"   Recall@5: {metrics.get('recall_at_5_rate', 0):.3f}")

        if metrics.get("mean_mrr"):
            print(f"   Mean Reciprocal Rank: {metrics['mean_mrr']:.3f}")

    if metrics.get("mean_similarity"):
        print(f"\n📈 Similarity Scores:")
        print(
            f"   Mean top result similarity: {metrics['mean_similarity']:.3f} ± {metrics.get('std_similarity', 0):.3f}"
        )

    if metrics.get("mean_gt_similarity"):
        print(
            f"   Mean ground truth similarity: {metrics['mean_gt_similarity']:.3f} ± {metrics.get('std_gt_similarity', 0):.3f}"
        )

    print(f"\n⚡ Performance:")
    print(f"   Average latency: {metrics['average_latency']:.1f}ms")

    # Overall score
    if metrics["queries_with_ground_truth"] > 0:
        overall_score = (
            metrics.get("exact_match_rate", 0) * 0.4
            + metrics.get("recall_at_3_rate", 0) * 0.3
            + metrics.get("mean_mrr", 0) * 0.3
        )
        print(f"\n🏆 Overall Score: {overall_score:.3f}")

        # Interpretation
        if overall_score > 0.8:
            print("   🟢 Excellent performance!")
        elif overall_score > 0.6:
            print("   🟡 Good performance")
        elif overall_score > 0.4:
            print("   🟠 Moderate performance")
        else:
            print("   🔴 Needs improvement")


def save_detailed_results(results_list, output_file):
    """Save detailed results to JSON file"""
    with open(output_file, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\n💾 Detailed results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-tower search model")
    parser.add_argument(
        "--num_queries",
        type=int,
        default=100,
        help="Number of queries to test (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for detailed results",
    )
    parser.add_argument(
        "--use_unique_queries",
        action="store_true",
        help="Ensure all test queries are unique",
    )
    args = parser.parse_args()

    setup_logging()

    print(f"🚀 Starting evaluation with {args.num_queries} queries...")
    print(f"   API URL: {FASTAPI_URL}")

    # Test API connectivity
    try:
        response = requests.get(f"{FASTAPI_URL}/ping", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not responding properly: {response.status_code}")
            return
        print("✅ API connection successful")
    except:
        print("❌ Cannot connect to FastAPI server")
        return

    results_list = []

    with tqdm(total=args.num_queries, desc="Evaluating queries") as pbar:
        while len(results_list) < args.num_queries:
            # Get random query
            result = get_random_query()
            if not result:
                continue
            results_list.append(result)

            # Update progress bar with current stats
            if result and result.get("evaluation", {}).get("top_result_matches_gt"):
                pbar.set_postfix(
                    {
                        "matches": sum(
                            1
                            for r in results_list
                            if r
                            and r.get("evaluation", {}).get("top_result_matches_gt")
                        )
                    }
                )

            pbar.update(1)

            # Small delay to be nice to the server
            time.sleep(0.1)

    print(f"\n✅ Completed evaluation of {len(results_list)} queries")

    # Calculate and display metrics
    metrics = calculate_metrics(results_list)
    print_metrics(metrics)

    # Save detailed results
    save_detailed_results(results_list, args.output)

    return metrics


if __name__ == "__main__":
    main()
