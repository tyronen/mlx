import torch
import json
from pathlib import Path
from tqdm import tqdm
import sacrebleu
import itertools
from models import image_caption, image_caption_utils
from common import utils
import logging

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
    # Reuse CocoDataset to handle loading, but we'll process it manually
    ds = image_caption.CocoDataset(split="val", use_official_captions=True)

    # Group by image filename/ID
    image_to_captions = {}
    unique_files = set()

    for row in ds.captions:
        fname = row["file_name"]
        caption = row["text"]
        if fname not in image_to_captions:
            image_to_captions[fname] = []
        image_to_captions[fname].append(caption)
        unique_files.add(fname)

    # Select a subset
    selected_files = sorted(list(unique_files))[:limit]
    print(f"Selected {len(selected_files)} images for evaluation.")

    # Prepare dataset: list of (feature_tensor, [ref_captions])
    eval_data = []
    print(f"[DEBUG] Image: {selected_files[0]}")
    for fname in selected_files:
        feat = ds.image_features[fname].to(DEVICE)  # [1024] or similar
        refs = image_to_captions[fname]
        eval_data.append((feat, refs))

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
        use_custom_decoder=False,  # Assuming style fine-tuned model uses LoRA/base structure
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_config(model, data, config):
    refs = []
    preds = []

    # Run inference
    for i, (img_feat, captions) in tqdm(
        enumerate(data), leave=False, desc="Inferencing"
    ):
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)

        with torch.no_grad():
            proj_feats = model.image_projection(img_feat)
            pred = generate_caption_inference(model, proj_feats, config)

        # DEBUG: Print first prediction and reference to see if model works
        if i == 0:
            print(f"\n[DEBUG] Pred: '{pred}'")
            print(f"[DEBUG] Ref[0]: '{captions[0]}'")

        preds.append(pred)
        refs.append(captions)  # List of strings

    # Compute SacreBLEU
    # sacrebleu.corpus_bleu expects:
    # sys: list of N hypotheses
    # refs: list of list of references (transposed!) -> [[ref1_i for i in N], [ref2_i for i in N]...]

    # Transpose refs
    max_refs = max(len(r) for r in refs)
    transposed_refs = []
    for k in range(max_refs):
        # Get k-th ref for each image, or empty string if missing
        ref_k = [r[k] if k < len(r) else "" for r in refs]
        transposed_refs.append(ref_k)

    bleu = sacrebleu.corpus_bleu(preds, transposed_refs)
    return bleu.score


def main():
    # 1. Load Model
    model_path = "data/base_coco_model.pth"  # The Style Fine-tuned model
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    model = load_trained_model(model_path)

    # 2. Prepare Data
    eval_data = get_val_data(limit=50)  # 50 images for faster feedback loop

    # 3. Setup Metric (SacreBLEU handles this internally)

    # 4. Define Grid
    grid = {
        "beam_width": [1],
        "length_penalty": [0.6],
        "repetition_penalty": [1.5, 2.0, 2.5],
        "no_repeat_ngram_size": [0],
    }

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting sweep over {len(combinations)} configurations...")
    print("-" * 60)
    print(f"{'Beam':<5} {'LenPen':<8} {'RepPen':<8} {'NoRep':<6} | {'BLEU':<8}")
    print("-" * 60)

    best_score = -1
    best_cfg = None

    for cfg in combinations:
        score = evaluate_config(model, eval_data, cfg)

        print(
            f"{cfg['beam_width']:<5} {cfg['length_penalty']:<8.1f} {cfg['repetition_penalty']:<8.1f} {cfg['no_repeat_ngram_size']:<6} | {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_cfg = cfg

    print("-" * 60)
    print(f"🏆 Best Configuration: {best_cfg}")
    print(f"📈 Best BLEU: {best_score:.4f}")


if __name__ == "__main__":
    main()
