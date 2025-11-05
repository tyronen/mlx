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


def generate_caption(image_features, model, max_length=50):
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

            # Get next token (greedy decoding)
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

    # Generation parameters
    max_length = 50

    if st.session_state.image_url:
        # Load and display image
        image = load_image_from_url(st.session_state.image_url)

        if image is not None:
            st.write(f"Image URL: {st.session_state.image_url}")
            st.image(image)
            with st.spinner("Generating captions..."), torch.no_grad():

                image_features = encoder([image])

                flickr_caption = generate_caption(
                    image_features, flickr_model, max_length
                )
                coco_caption = generate_caption(image_features, coco_model, max_length)
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
