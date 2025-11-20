import streamlit as st
import torch
import requests
from PIL import Image
import io
import logging
from models import image_caption, image_caption_utils
import time
from common import utils

# Suppress warnings
utils.setup_logging()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration - update these paths as needed
DEVICE = utils.get_device()


def load_model(dataset, custom):
    filename = (
        image_caption_utils.CUSTOM_FLICKR_MODEL_FILE
        if dataset == "flickr" and custom
        else (
            image_caption_utils.CUSTOM_COCO_MODEL_FILE
            if dataset == "coco" and custom
            else (
                (
                    image_caption_utils.BASE_FLICKR_MODEL_FILE
                    if dataset == "flickr"
                    else image_caption_utils.BASE_COCO_MODEL_FILE
                )
            )
        )
    )
    checkpoint = torch.load(filename, map_location=DEVICE)
    model = image_caption.CombinedTransformer(
        model_dim=checkpoint["model_dim"],
        ffn_dim=checkpoint["ffn_dim"],
        num_heads=checkpoint["num_heads"],
        num_decoders=checkpoint["num_decoders"],
        dropout=checkpoint["dropout"],
        use_custom_decoder=custom,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_models():
    """Load the trained model"""
    encoder = image_caption.ImageEncoder()
    flickr_model = load_model(dataset="flickr", custom=False)
    coco_model = load_model(dataset="coco", custom=False)
    st.success("Model loaded successfully!")
    return encoder, flickr_model, coco_model


@st.cache_data
def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0, min_tokens_to_keep=1):
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


def generate_caption(
    image_features,
    model,
    max_length=32,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
):
    """Generate caption for the image"""
    try:
        # Encode image
        image_features = model.image_projection(image_features)  # [1, model_dim]

        # Initialize with BOS token
        generated = [model.tokenizer.bos_token_id]

        # Generate tokens one by one
        for _ in range(max_length):
            # Convert to tensor
            input_ids = torch.tensor([generated], device=DEVICE)

            # Use the new decode_step method to get the next token's logits
            logits = model.decode_step(image_features, input_ids)

            # Apply repetition penalty
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

            # Enforce no-repeat n-gram constraint
            banned = _get_banned_next_tokens(generated, no_repeat_ngram_size)
            if banned:
                logits[0, torch.tensor(banned, device=logits.device)] = -float("inf")

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
        return caption

    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Error generating caption"


def main():
    st.title("🖼️ Image Captioning Server")

    # Load model
    encoder, flickr_model, coco_model = load_models()

    # Initialize session state variables if they don't exist
    if "image_url" not in st.session_state:
        st.session_state.image_url = ""
    if "flickr_caption" not in st.session_state:
        st.session_state.flickr_caption = ""

    if st.button("Random image 🚀"):
        # Cache-buster so you don’t get the same photo twice
        seed = int(time.time() * 1000)  # or random.randint(0, 1e9)
        st.session_state.image_url = f"https://picsum.photos/seed/{seed}/640/480"
        st.session_state.flickr_caption = ""
        st.session_state.coco_caption = ""
        st.rerun()

    # Decoding parameters
    with st.expander("Decoding settings"):
        c1, c2, c3 = st.columns(3)
        temperature = c1.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        top_p = c2.slider("Top-p (nucleus)", 0.0, 1.0, 0.9, 0.01)
        top_k = c3.slider("Top-k (0 = off)", 0, 200, 50, 1)
        c4, c5 = st.columns(2)
        repetition_penalty = c4.slider("Repetition penalty", 1.0, 2.0, 1.2, 0.05)
        no_repeat_ngram_size = c5.slider("No-repeat n-gram size", 0, 5, 3, 1)
        max_length = st.slider("Max length", 8, 64, 32, 1)

    if st.session_state.image_url:
        # Load and display image
        image = load_image_from_url(st.session_state.image_url)

        if image is not None:
            st.write(f"Image URL: {st.session_state.image_url}")
            st.image(image)
            with st.spinner("Generating captions..."), torch.no_grad():

                image_features = encoder([image])

                flickr_caption = generate_caption(
                    image_features,
                    flickr_model,
                    max_length,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    no_repeat_ngram_size,
                )
                coco_caption = generate_caption(
                    image_features,
                    coco_model,
                    max_length,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty,
                    no_repeat_ngram_size,
                )
                st.session_state.flickr_caption = flickr_caption
                st.session_state.coco_caption = coco_caption

            st.markdown(f"**Flickr caption:** {st.session_state.flickr_caption}")
            st.markdown(f"**COCO caption:** {st.session_state.coco_caption}")

        else:
            st.error("Failed to load image from URL")
    else:
        st.info("👆 Enter an image URL to get started")


if __name__ == "__main__":
    main()
