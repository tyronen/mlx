import torch
from pathlib import Path
from tqdm import tqdm
import sacrebleu
import statistics
import itertools
from PIL import Image
import os
import argparse
from models import image_caption
from common import utils

# Setup
utils.setup_logging()
DEVICE = utils.get_device()
print(f"Using device: {DEVICE}")

# --- Reimplement Generation Logic (avoiding Streamlit deps) ---
# Copied and adapted from ui/image_caption_page.py to ensure identical behavior


def _normalized_beam_score(score, length, length_penalty):
    if length_penalty is None or length_penalty <= 0:
        return score
    length = max(1, length)
    return score / (length**length_penalty)


def _get_banned_next_tokens(generated, no_repeat_ngram_size):
    if no_repeat_ngram_size is None or no_repeat_ngram_size <= 0:
        return []
    n = no_repeat_ngram_size
    if len(generated) + 1 < n:
        return []
    ngrams = {}
    for i in range(len(generated) - n + 1):
        prefix = tuple(generated[i : i + n - 1])
        nxt = generated[i + n - 1]
        if prefix not in ngrams:
            ngrams[prefix] = set()
        ngrams[prefix].add(nxt)
    current_prefix = tuple(generated[-(n - 1) :])
    return list(ngrams.get(current_prefix, []))


def _apply_decoding_constraints(
    logits, generated, repetition_penalty, no_repeat_ngram_size
):
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits = logits.clone()
        ids = torch.tensor(generated, device=logits.device)
        unique_ids = torch.unique(ids)
        selected = logits[0, unique_ids]
        adjusted = torch.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
        logits[0, unique_ids] = adjusted

    banned = _get_banned_next_tokens(generated, no_repeat_ngram_size)
    if banned:
        logits[0, torch.tensor(banned, device=logits.device)] = -float("inf")
    return logits


def _beam_search_decode(
    image_features,
    model,
    max_length,
    repetition_penalty,
    no_repeat_ngram_size,
    beam_width,
    length_penalty,
):
    eos = model.tokenizer.eos_token_id
    # Use prompt to guide beam search too
    prompt_tokens = model.tokenizer.encode("A photo of", add_special_tokens=False)
    start_seq = [model.tokenizer.bos_token_id] + prompt_tokens
    beams = [(start_seq, 0.0, False)]
    completed = []

    for _ in range(max_length):
        all_candidates = []
        for tokens, score, finished in beams:
            if finished:
                all_candidates.append((tokens, score, True))
                continue

            input_ids = torch.tensor([tokens], device=DEVICE)
            logits = model.decode_step(image_features, input_ids)
            logits = _apply_decoding_constraints(
                logits, tokens, repetition_penalty, no_repeat_ngram_size
            )

            log_probs = torch.log_softmax(logits[0], dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_width, dim=-1)
            for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_tokens = tokens + [idx]
                new_score = score + float(log_prob)
                all_candidates.append((new_tokens, new_score, idx == eos))

        if not all_candidates:
            break

        all_candidates.sort(
            key=lambda item: _normalized_beam_score(
                item[1], len(item[0]) - 1, length_penalty
            ),
            reverse=True,
        )
        beams = all_candidates[:beam_width]

        next_beams = []
        for tokens, score, finished in beams:
            if finished:
                completed.append((tokens, score))
            else:
                next_beams.append((tokens, score, False))
        beams = next_beams
        if not beams and completed:
            break

    if completed:
        completed.sort(
            key=lambda item: _normalized_beam_score(
                item[1], len(item[0]) - 1, length_penalty
            ),
            reverse=True,
        )
        return completed[0][0]
    if beams:
        return beams[0][0]
    return [model.tokenizer.bos_token_id]


def generate_caption_inference(model, image_features, config):
    # Unpack config
    beam_width = config.get("beam_width", 1)

    if beam_width > 1:
        token_ids = _beam_search_decode(
            image_features,
            model,
            max_length=config.get("max_length", 32),
            repetition_penalty=config.get("repetition_penalty", 1.0),
            no_repeat_ngram_size=config.get("no_repeat_ngram_size", 0),
            beam_width=beam_width,
            length_penalty=config.get("length_penalty", 1.0),
        )
        return model.tokenizer.decode(token_ids[1:], skip_special_tokens=True)
    else:
        # Simple greedy/sampling path (simplified for this sweep as we care mostly about beam)
        token_ids = _beam_search_decode(
            image_features,
            model,
            config.get("max_length", 32),
            config.get("repetition_penalty", 1.0),
            config.get("no_repeat_ngram_size", 0),
            1,
            config.get("length_penalty", 1.0),
        )
        return model.tokenizer.decode(token_ids[1:], skip_special_tokens=True)


# --- Data Preparation ---


def get_val_data(limit=200):
    """Load validation features and group ground truth captions by image."""
    print("Loading validation dataset...")
    # Reuse CocoDataset to handle loading. With `use_official_captions=True`,
    # the dataset groups captions per image and stores them in `ds.data` as
    # (filename, [captions]) tuples, so we don't need to iterate over __getitem__
    # (which returns tensors).
    ds = image_caption.CocoDataset(split="val", use_official_captions=True)

    # Select a subset (dataset ordering is deterministic due to seed in dataset)
    selected = ds.data[:limit]
    print(f"Selected {len(selected)} images for evaluation.")

    # Prepare dataset: list of (image_path, [ref_captions])
    eval_data = []
    if selected:
        print(f"[DEBUG] Image: {selected[0][0]}")
    for fname, refs in selected:
        image_path = os.path.join(ds.image_dir, fname)
        eval_data.append((image_path, refs))

    return eval_data


# --- Main Sweep Logic ---


def load_trained_model(model_path):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = image_caption.CombinedTransformer(
        model_dim=checkpoint["model_dim"],
        ffn_dim=checkpoint["ffn_dim"],
        num_heads=checkpoint["num_heads"],
        num_decoders=checkpoint["num_decoders"],
        dropout=checkpoint["dropout"],
        use_mlp_projector=checkpoint.get("use_mlp_projector", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_config(model, vision_encoder, data, config):
    refs = []
    preds = []

    # Run inference
    for i, (image_path, captions) in tqdm(
        enumerate(data), leave=False, desc="Inferencing", total=len(data)
    ):
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            inputs = vision_encoder.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(DEVICE)

            with torch.no_grad():
                # Encode image [B, 257, 1024]
                img_feat = vision_encoder(pixel_values)
                # Project and generate
                proj_feats = model.image_projection(img_feat)
                pred = generate_caption_inference(model, proj_feats, config)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            pred = ""

        preds.append(pred)
        refs.append(captions)  # List of strings

    # Compute SacreBLEU
    max_refs = max(len(r) for r in refs)
    transposed_refs = []
    for k in range(max_refs):
        ref_k = [r[k] if k < len(r) else "" for r in refs]
        transposed_refs.append(ref_k)

    bleu = sacrebleu.corpus_bleu(preds, transposed_refs)
    return bleu.score


def summarize_hparam_impacts(results, grid_keys):
    """Aggregate BLEU scores per hyperparameter value to show impact trends."""
    print("\nHyperparameter impact (mean/median/best BLEU per value)")
    for key in grid_keys:
        buckets = {}
        for cfg, score in results:
            buckets.setdefault(cfg[key], []).append(score)

        summary = []
        for val, scores in buckets.items():
            if not scores:
                continue
            summary.append(
                (
                    val,
                    statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    statistics.mean(scores),
                    statistics.median(scores),
                    max(scores),
                    min(scores),
                    len(scores),
                )
            )
        # Sort by MEAN (index 2) descending to show best performers first
        summary.sort(key=lambda x: x[2], reverse=True)

        print("-" * 80)
        print(f"{key}:")
        print(
            f"{'Value':<10} {'Mean':<10} {'Median':<10} {'Best':<10} {'Stdev':<10} {'Worst':<10} {'N':<5}"
        )
        for (
            val,
            std_score,
            mean_score,
            median_score,
            best_score,
            worst_score,
            count,
        ) in summary:
            print(
                f"{str(val):<10} {mean_score:<10.4f} {median_score:<10.4f} "
                f"{best_score:<10.4f} {std_score:<10.4f} {worst_score:<10.4f} {count:<5}"
            )
        if summary:
            best_val = summary[0][0]
            print(f"Best {key} (by mean BLEU): {best_val}")
    print("-" * 80)


def get_args():
    parser = argparse.ArgumentParser(description="Optimize inference hyperparameters")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of images to evaluate (default: 30). Increase to 100+ for stable results.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/base_coco_model.pth",
        help="Path to model",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Beam widths to sweep",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        nargs="+",
        default=[0.6],
        help="Length penalties to sweep",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        nargs="+",
        default=[1.0],
        help="Repetition penalties to sweep",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        nargs="+",
        default=[0],
        help="No repeat ngram sizes to sweep",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # 1. Load Model
    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    model = load_trained_model(model_path)

    # Load Vision Encoder
    vision_encoder = image_caption.ImageEncoder().to(DEVICE)
    vision_encoder.eval()

    # 2. Prepare Data
    eval_data = get_val_data(limit=args.limit)

    # 4. Define Grid
    grid = {
        "beam_width": args.beam_width,
        "length_penalty": args.length_penalty,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(
        f"Starting sweep over {len(combinations)} configurations with limit={args.limit}..."
    )
    print("-" * 80)
    print(f"{'Beam':<5} {'LenPen':<8} {'RepPen':<8} {'NoRep':<6} | {'BLEU':<8}")
    print("-" * 80)

    best_score = -1
    best_cfg = None
    results = []

    for cfg in combinations:
        score = evaluate_config(model, vision_encoder, eval_data, cfg)
        results.append((cfg, score))

        print(
            f"{cfg['beam_width']:<5} {cfg['length_penalty']:<8.1f} {cfg['repetition_penalty']:<8.1f} {cfg['no_repeat_ngram_size']:<6} | {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_cfg = cfg

    print("-" * 80)
    print(f"🏆 Best Configuration: {best_cfg}")
    print(f"📈 Best BLEU: {best_score:.4f}")
    summarize_hparam_impacts(results, keys)


if __name__ == "__main__":
    main()
