import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


def analyze_alignment_heads(json_file_path: str, model_name: str = "large-v3"):
    """
    Analyze alignment heads performance from CrisperWhisper head_results.json

    Args:
        json_file_path: Path to the head_results.json file
        model_name: Model to analyze (e.g., "large-v3", "large-v3-turbo")
    """

    # Load the JSON data
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Filter for the specified model
    model_data = [entry for entry in data if entry.get("model") == model_name]

    if not model_data:
        print(f"No data found for model: {model_name}")
        available_models = set(entry.get("model") for entry in data)
        print(f"Available models: {available_models}")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(model_data)

    # Extract layer and head numbers
    df["layer"] = df["heads"].apply(lambda x: x[0][0] if len(x) == 1 else None)
    df["head"] = df["heads"].apply(lambda x: x[0][1] if len(x) == 1 else None)
    df["head_str"] = df["heads"].apply(lambda x: str(x[0]) if len(x) == 1 else str(x))
    df["num_heads"] = df["heads"].apply(len)

    # Filter for single heads only (for initial analysis)
    single_heads = df[df["num_heads"] == 1].copy()

    print(f"\n=== Analysis for {model_name} ===")
    print(f"Total entries: {len(df)}")
    print(f"Single head entries: {len(single_heads)}")
    print(f"Multi-head combinations: {len(df) - len(single_heads)}")

    # Basic statistics
    f1_col = "F1_collar.0.1"
    iou_col = "avg_iou_collar.0.1"

    print(f"\n=== Performance Statistics (Single Heads) ===")
    print(
        f"F1-score range: {single_heads[f1_col].min():.6f} - {single_heads[f1_col].max():.6f}"
    )
    print(
        f"IoU range: {single_heads[iou_col].min():.6f} - {single_heads[iou_col].max():.6f}"
    )
    print(f"F1-score mean: {single_heads[f1_col].mean():.6f}")
    print(f"IoU mean: {single_heads[iou_col].mean():.6f}")

    # Top performers by F1-score
    print(f"\n=== Top 20 Single Heads by F1-Score ===")
    top_f1 = single_heads.nlargest(20, f1_col)
    for idx, row in top_f1.iterrows():
        print(
            f"[{row['layer']}, {row['head']}]: F1={row[f1_col]:.6f}, IoU={row[iou_col]:.6f}"
        )

    # Top performers by IoU
    print(f"\n=== Top 20 Single Heads by IoU ===")
    top_iou = single_heads.nlargest(20, iou_col)
    for idx, row in top_iou.iterrows():
        print(
            f"[{row['layer']}, {row['head']}]: F1={row[f1_col]:.6f}, IoU={row[iou_col]:.6f}"
        )

    # Combined score (you can adjust the weighting)
    single_heads["combined_score"] = (
        0.7 * single_heads[f1_col] + 0.3 * single_heads[iou_col]
    )

    print(f"\n=== Top 20 Single Heads by Combined Score (0.7*F1 + 0.3*IoU) ===")
    top_combined = single_heads.nlargest(20, "combined_score")
    for idx, row in top_combined.iterrows():
        print(
            f"[{row['layer']}, {row['head']}]: F1={row[f1_col]:.6f}, IoU={row[iou_col]:.6f}, Combined={row['combined_score']:.6f}"
        )

    # Performance by layer
    print(f"\n=== Performance by Layer (Single Heads) ===")
    layer_stats = (
        single_heads.groupby("layer")
        .agg({f1_col: ["mean", "max"], iou_col: ["mean", "max"], "head": "count"})
        .round(6)
    )
    print(layer_stats)

    # Find optimal thresholds
    f1_95th = np.percentile(single_heads[f1_col], 95)
    f1_90th = np.percentile(single_heads[f1_col], 90)
    iou_95th = np.percentile(single_heads[iou_col], 95)
    iou_90th = np.percentile(single_heads[iou_col], 90)

    print(f"\n=== Performance Thresholds ===")
    print(f"F1-score 95th percentile: {f1_95th:.6f}")
    print(f"F1-score 90th percentile: {f1_90th:.6f}")
    print(f"IoU 95th percentile: {iou_95th:.6f}")
    print(f"IoU 90th percentile: {iou_90th:.6f}")

    # High-performing heads (above 90th percentile in either metric)
    high_performers = single_heads[
        (single_heads[f1_col] >= f1_90th) | (single_heads[iou_col] >= iou_90th)
    ].sort_values("combined_score", ascending=False)

    print(f"\n=== High-Performing Heads (90th percentile+) ===")
    print(f"Count: {len(high_performers)}")

    alignment_heads = []
    for idx, row in high_performers.iterrows():
        alignment_heads.append([int(row["layer"]), int(row["head"])])
        print(
            f"[{row['layer']}, {row['head']}]: F1={row[f1_col]:.6f}, IoU={row[iou_col]:.6f}"
        )

    # Multi-head combinations analysis
    multi_heads = df[df["num_heads"] > 1]
    if len(multi_heads) > 0:
        print(f"\n=== Top Multi-Head Combinations ===")
        top_multi = multi_heads.nlargest(10, f1_col)
        for idx, row in top_multi.iterrows():
            print(f"{row['head_str']}: F1={row[f1_col]:.6f}, IoU={row[iou_col]:.6f}")

        # Find the best overall combination
        best_overall = multi_heads.loc[multi_heads[f1_col].idxmax()]
        print(f"\n=== Best Overall Combination ===")
        print(f"Heads: {best_overall['heads']}")
        print(f"F1-Score: {best_overall[f1_col]:.6f}")
        print(f"IoU: {best_overall[iou_col]:.6f}")

        optimal_alignment_heads = best_overall["heads"]
    else:
        # If no multi-head combinations, suggest top single heads
        optimal_alignment_heads = alignment_heads[:10]  # Top 10 single heads

    # Output for use in Whisper
    print(f"\n=== Recommended Alignment Heads for {model_name} ===")
    print(f"alignment_heads = {optimal_alignment_heads}")

    # Save results to files
    output_file = f"data/{model_name}_alignment_analysis.csv"
    single_heads.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save recommended heads
    recommendations = {
        "model": model_name,
        "recommended_heads": optimal_alignment_heads,
        "top_single_heads": alignment_heads[:20],
        "performance_thresholds": {
            "f1_90th": f1_90th,
            "f1_95th": f1_95th,
            "iou_90th": iou_90th,
            "iou_95th": iou_95th,
        },
    }

    with open(f"data/{model_name}_recommended_heads.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    return single_heads, multi_heads, recommendations


def compare_models(
    json_file_path: str, models: List[str] = ["large-v3", "large-v3-turbo"]
):
    """Compare performance across different models"""

    with open(json_file_path, "r") as f:
        data = json.load(f)

    print("=== Model Comparison ===")

    for model in models:
        model_data = [
            entry
            for entry in data
            if entry.get("model") == model and len(entry.get("heads", [])) == 1
        ]
        if model_data:
            df = pd.DataFrame(model_data)
            f1_mean = df["F1_collar.0.1"].mean()
            iou_mean = df["avg_iou_collar.0.1"].mean()
            f1_max = df["F1_collar.0.1"].max()
            iou_max = df["avg_iou_collar.0.1"].max()

            print(f"\n{model}:")
            print(f"  F1 - Mean: {f1_mean:.6f}, Max: {f1_max:.6f}")
            print(f"  IoU - Mean: {iou_mean:.6f}, Max: {iou_max:.6f}")
            print(f"  Total single heads: {len(model_data)}")


if __name__ == "__main__":
    # Usage example
    json_file_path = "data/head_results.json"  # Update this path

    # Analyze large-v3
    print("Analyzing whisper-large-v3...")
    single_heads_v3, multi_heads_v3, recs_v3 = analyze_alignment_heads(
        json_file_path, "large-v3"
    )

    # Analyze large-v3-turbo if available
    print("\n" + "=" * 80)
    print("Analyzing whisper-large-v3-turbo...")
    try:
        single_heads_turbo, multi_heads_turbo, recs_turbo = analyze_alignment_heads(
            json_file_path, "large-v3-turbo"
        )
    except:
        print("No data found for large-v3-turbo")

    # Compare models
    print("\n" + "=" * 80)
    compare_models(json_file_path)
