import json
import os
import random
import re
import tempfile
import zipfile
import json
from pathlib import Path

import requests
import streamlit as st
import torch
from PIL import Image
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import image_caption, image_caption_utils
from common import utils

# Suppress warnings
utils.setup_logging()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration - update these paths as needed
DEVICE = utils.get_device()
COCO_TEST_DIR_CANDIDATES = [
    Path("data/coco"),
]
COCO_TEST_ZIP_URL = "http://images.cocodataset.org/zips/test2017.zip"
COCO_TEST_ZIP_NAME = "test2017.zip"


def load_model(model_type="base"):
    filename = (
        image_caption_utils.OFFICIAL_COCO_MODEL_FILE
        if model_type == "official"
        else image_caption_utils.BASE_COCO_MODEL_FILE
    )

    checkpoint = torch.load(filename, map_location=DEVICE)
    model = image_caption.CombinedTransformer(
        model_dim=checkpoint["model_dim"],
        ffn_dim=checkpoint["ffn_dim"],
        num_heads=checkpoint["num_heads"],
        num_decoders=checkpoint["num_decoders"],
        dropout=checkpoint["dropout"],
        use_custom_decoder=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_grammar_model():
    """Load the grammar refinement model (Qwen2.5-0.5B-Instruct)"""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Load in bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    return tokenizer, model


def refine_caption_grammar(tokenizer, model, draft_caption):
    """Use the instruct model to fix grammar"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that fixes grammar. "
                "Rewrite the user's text as a single, natural English sentence "
                "describing an image. Do not add new information."
            ),
        },
        {"role": "user", "content": draft_caption},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=64, temperature=0.3, top_p=0.9
        )

    # Decode only the new tokens
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


@st.cache_resource
def load_models(model_type="base"):
    """Load the trained model"""
    encoder = image_caption.ImageEncoder()
    coco_model = load_model(model_type=model_type)
    st.success(f"Loaded {model_type} model successfully!")
    return encoder, coco_model


@st.cache_data(show_spinner=False)
def load_coco_test_files():
    """Locate COCO test images on disk and cache the file list."""
    valid_exts = {".jpg", ".jpeg", ".png"}
    for directory in COCO_TEST_DIR_CANDIDATES:
        if not directory.exists():
            continue
        image_files = sorted(
            [p for p in directory.iterdir() if p.suffix.lower() in valid_exts]
        )
        if image_files:
            return directory, image_files
    return None, []


def _download_and_extract_coco_test():
    data_root = Path("data")
    data_root.mkdir(parents=True, exist_ok=True)

    status = st.status(
        "Downloading COCO test2017 (~6GB). This runs once.", state="running"
    )
    progress = st.progress(0.0)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path = Path(tmp_file.name)
    try:
        with requests.get(COCO_TEST_ZIP_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                tmp_file.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress.progress(min(downloaded / total, 1.0))
        tmp_file.close()
        status.update(label="Extracting COCO test2017 images...", state="running")
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            zip_ref.extractall(data_root)
        status.update(label="COCO test2017 ready ✅", state="complete")
        progress.progress(1.0)
    except Exception as exc:
        status.update(label=f"Download failed: {exc}", state="error")
        raise
    finally:
        tmp_file.close()
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def ensure_coco_test_data():
    directory, files = load_coco_test_files()
    if files:
        return directory, files
    _download_and_extract_coco_test()
    load_coco_test_files.clear()
    directory, files = load_coco_test_files()
    if not files:
        raise RuntimeError("COCO test dataset download failed.")
    return directory, files


def _top_k_top_p_filtering(logits, top_k, top_p, min_tokens_to_keep=1):
    logits = logits.clone()
    vocab = logits.size(-1)
    if top_k and top_k > 0:
        k = min(max(top_k, min_tokens_to_keep), vocab)
        thresh = torch.topk(logits, k)[0][..., -1, None]
        remove = logits < thresh
        logits[remove] = -float("inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 0:
            sorted_remove[..., :min_tokens_to_keep] = False
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        logits[remove] = -float("inf")
    return logits


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


def _normalized_beam_score(score, length, length_penalty):
    if length_penalty is None or length_penalty <= 0:
        return score
    length = max(1, length)
    return score / (length**length_penalty)


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


def _capitalize_sentences(text):
    text = text.strip()
    if not text:
        return text
    parts = re.split(r"([.!?]+)", text)
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()
        if not sentence:
            continue
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        sentence = sentence[0].upper() + sentence[1:]
        sentences.append((sentence, punct.strip()))
    result = []
    for sentence, punct in sentences:
        if punct:
            result.append(f"{sentence}{punct}")
        else:
            result.append(sentence)
    return " ".join(result).strip()


def generate_caption(
    image_features,
    model,
    max_length,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    no_repeat_ngram_size,
    beam_width,
    length_penalty,
):
    """Generate caption for the image"""
    try:
        # Encode image
        image_features = model.image_projection(image_features)  # [1, model_dim]

        if beam_width and beam_width > 1:
            token_ids = _beam_search_decode(
                image_features,
                model,
                max_length,
                repetition_penalty,
                no_repeat_ngram_size,
                beam_width,
                length_penalty,
            )
            caption = model.tokenizer.decode(token_ids[1:], skip_special_tokens=True)
            caption = _capitalize_sentences(caption)
            return caption

        # Initialize with BOS token + prompt to guide grammar
        # "A photo of" helps ground the decoder into descriptive English mode
        prompt_tokens = model.tokenizer.encode("A photo of", add_special_tokens=False)
        generated = [model.tokenizer.bos_token_id] + prompt_tokens

        # Generate tokens one by one
        for _ in range(max_length):
            # Convert to tensor
            input_ids = torch.tensor([generated], device=DEVICE)

            # Use the new decode_step method to get the next token's logits
            logits = model.decode_step(image_features, input_ids)

            logits = _apply_decoding_constraints(
                logits, generated, repetition_penalty, no_repeat_ngram_size
            )

            # Temperature and sampling
            if temperature is not None and temperature > 0:
                logits = logits / temperature
                filt = _top_k_top_p_filtering(
                    logits[0],
                    top_k=(top_k or 0),
                    top_p=(top_p if top_p is not None else 1.0),
                ).unsqueeze(0)
                probs = torch.softmax(filt, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy
                next_token = torch.argmax(logits, dim=-1).item()

            # Stop if EOS token
            if next_token == model.tokenizer.eos_token_id:
                break

            generated.append(next_token)

        # Decode the generated tokens
        caption = model.tokenizer.decode(generated[1:], skip_special_tokens=True)
        caption = _capitalize_sentences(caption)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Error generating caption"


def official_caption(target):
    data = json.load(open("data/annotations/captions_train2017.json"))
    by_file_name = {img["file_name"]: img["id"] for img in data["images"]}
    st.write(target)
    image_id = by_file_name.get(target)
    st.write(image_id)
    captions = {}
    for ann in data["annotations"]:
        captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return captions.get(image_id, [])


def main():
    st.title("🖼️ Image Captioning Server")

    # Load model and ensure COCO test images are present
    try:
        coco_dir, coco_files = ensure_coco_test_data()
    except Exception as exc:
        st.error(f"Unable to prepare COCO test dataset: {exc}")
        return

    if not coco_files:
        st.error(
            "COCO test images not found. Please download them "
            "to one of: data/coco_test2017, data/coco_test, data/coco/test2017, data/coco."
        )
        return

    model_choice = st.sidebar.radio(
        "Model Version",
        ["Style Fine-tuned", "Foundation (Official)"],
        index=0,
        help="Switch between the style-adapted model and the foundation model trained on official captions.",
    )
    model_type = "base" if model_choice == "Style Fine-tuned" else "official"

    encoder, coco_model = load_models(model_type=model_type)

    # Initialize session state variables if they don't exist
    if "coco_test_path" not in st.session_state:
        st.session_state.coco_test_path = ""
    if "coco_caption" not in st.session_state:
        st.session_state.coco_caption = ""
    if "correct_caption" not in st.session_state:
        st.session_state.correct_caption = []

    # Settings form
    with st.sidebar.form("generation_settings"):
        st.subheader("Decoding Settings")
        c1, c2 = st.columns(2)
        temperature = c1.slider("Temperature", 0.0, 2.0, 0.0, 0.05)
        top_p = c2.slider("Top-p (nucleus)", 0.0, 1.0, 1.0, 0.01)

        c3, c4 = st.columns(2)
        top_k = c3.slider("Top-k (0 = off)", 0, 200, 0, 1)
        repetition_penalty = c4.slider("Repetition penalty", 1.0, 2.0, 1.5, 0.05)

        c5, c6 = st.columns(2)
        no_repeat_ngram_size = c5.slider("No-repeat n-gram", 0, 5, 4, 1)
        max_length = c6.slider("Max length", 8, 64, 24, 1)

        c7, c8 = st.columns(2)
        beam_width = c7.slider("Beam width", 1, 8, 1, 1)
        length_penalty = c8.slider("Length penalty", 0.0, 1.5, 0.6, 0.05)

        if beam_width > 1:
            st.caption("Beam search active (overrides temp/top-k).")
        else:
            st.caption("Sampling mode active.")

        st.divider()
        enable_grammar_refinement = st.checkbox(
            "✨ Refine Grammar (Beta)",
            value=True,
            help="Uses a tiny LLM to rewrite the output into natural English.",
        )

        submitted = st.form_submit_button("Apply Settings & Generate")

    # Image selection (outside form so it updates immediately)
    if st.button("Random test image 🚀"):
        st.session_state.coco_test_path = str(random.choice(coco_files))
        # Clear previous captions when image changes
        st.session_state.coco_caption = ""
        st.session_state.correct_caption = []
        # We want to trigger generation immediately for a new image
        submitted = True

    image_path = st.session_state.coco_test_path
    if image_path:
        image = Image.open(image_path).convert("RGB")
        st.write(f"Image path: {Path(image_path).name} (from {coco_dir})")
        st.image(image)

        # Only generate if button pressed or new image selected
        if submitted or (st.session_state.coco_caption == "" and image_path):
            with st.spinner("Generating captions..."), torch.no_grad():
                # Preprocess image
                inputs = encoder.processor(images=image, return_tensors="pt")
                pixel_values = inputs.pixel_values
                # Run encoder
                image_features = encoder(pixel_values)

                coco_caption = generate_caption(
                    image_features,
                    coco_model,
                    max_length,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    no_repeat_ngram_size,
                    beam_width,
                    length_penalty,
                )

                if enable_grammar_refinement:
                    with st.spinner("Refining grammar..."):
                        cleaner_tokenizer, cleaner_model = load_grammar_model()
                        coco_caption = refine_caption_grammar(
                            cleaner_tokenizer, cleaner_model, coco_caption
                        )

                correct_caption = official_caption(image_path.split("/")[-1])
                st.session_state.coco_caption = coco_caption
                st.session_state.correct_caption = correct_caption

        if st.session_state.coco_caption:
            st.markdown(f"**Caption:** {st.session_state.coco_caption}")
        if st.session_state.correct_caption:
            st.markdown(f"**Correct caption:** {st.session_state.correct_caption}")
    else:
        st.info("👆 Click “Random test image” to caption a sample.")


if __name__ == "__main__":
    main()
