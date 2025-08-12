import streamlit as st
import torch
import requests
from PIL import Image
import io
import logging
import models
import time
import utils

# Suppress warnings
utils.setup_logging()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration - update these paths as needed
DEVICE = utils.get_device()


def load_model(custom):
    filename = utils.CUSTOM_MODEL_FILE if custom else utils.BASE_MODEL_FILE
    checkpoint = torch.load(filename, map_location=DEVICE)
    model = models.CombinedTransformer(
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
    vit_encoder = models.VitEncoder()
    custom_model = load_model(custom=True)
    base_model = load_model(custom=False)
    st.success("Model loaded successfully!")
    return vit_encoder, custom_model, base_model


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
    st.set_page_config(
        page_title="Image Captioning Server", page_icon="🖼️", layout="wide"
    )

    st.title("🖼️ Image Captioning Server")

    # Load model
    vit_encoder, custom_model, base_model = load_models()

    # Initialize session state variables if they don't exist
    if "image_url" not in st.session_state:
        st.session_state.image_url = ""
    if "custom_caption" not in st.session_state:
        st.session_state.custom_caption = ""

    if st.button("Random image 🚀"):
        # Cache-buster so you don’t get the same photo twice
        seed = int(time.time() * 1000)  # or random.randint(0, 1e9)
        st.session_state.image_url = f"https://picsum.photos/seed/{seed}/640/480"
        st.session_state.custom_caption = ""
        st.rerun()

    # Generation parameters
    max_length = 50

    if st.session_state.image_url:
        # Load and display image
        image = load_image_from_url(st.session_state.image_url)

        if image is not None:
            with st.spinner("Generating captions..."), torch.no_grad():

                image_features = vit_encoder([image])

                custom_caption = generate_caption(
                    image_features, custom_model, max_length
                )
                base_caption = generate_caption(image_features, base_model, max_length)
                st.session_state.custom_caption = custom_caption
                st.session_state.base_caption = base_caption

            st.image(image)
            st.write(st.session_state.custom_caption)
            st.write(st.session_state.base_caption)

        else:
            st.error("Failed to load image from URL")
    else:
        st.info("👆 Enter an image URL to get started")


if __name__ == "__main__":
    main()
