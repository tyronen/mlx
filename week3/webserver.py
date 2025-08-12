import inspect
import random
import string

import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

import models
import utils

device = utils.get_device()


@st.cache_resource
def load_model():
    checkpoint = torch.load(utils.COMPLEX_MODEL_FILE, map_location=device)
    config = checkpoint["config"]

    # keep only the arguments that ComplexTransformer’s __init__ expects
    ctor_keys = inspect.signature(models.ComplexTransformer).parameters
    ctor_cfg = {k: v for k, v in config.items() if k in ctor_keys}

    model = models.ComplexTransformer(**ctor_cfg)

    # Strip torch.compile prefixes if present
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod.") :]
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)  # keep parameters on the same device as inputs
    model.eval()
    return model


def preprocess_image(image_data):
    """Preprocess one 280×280 canvas → (1, 28, 28) normalised tensor."""
    image = Image.fromarray(image_data).convert("L")

    # tight crop around the drawing
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # centre‑pad to square then resize to 28×28
    w, h = image.size
    side = max(w, h) + 40
    pad = (
        (side - w) // 2,
        (side - h) // 2,
        side - w - (side - w) // 2,
        side - h - (side - h) // 2,
    )
    image = ImageOps.expand(image, border=pad, fill=0)

    tfm = transforms.Compose(
        [
            transforms.Resize(
                (28, 28),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
    return tfm(image).squeeze(0)  # (28, 28)


INTRO = """
This is a demonstration app showing simple handwriting recognition.

This app uses a deep-learning model trained on the MNIST public dataset of 
handwritten digits using the Pytorch library.

Draw digits (0-9) in the black boxes and press Predict. The model will then 
attempt to guess what digits you have entered."""


def assemble_composite(tl, tr, bl, br):
    """Stack four (28,28) tensors into one (1,56,56) composite."""
    composite = torch.zeros(56, 56)
    composite[:28, :28] = tl
    composite[:28, 28:] = tr
    composite[28:, :28] = bl
    composite[28:, 28:] = br
    return composite.unsqueeze(0).unsqueeze(0)  # (1,1,56,56)


def random_string():
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )


def make_canvas(key_index):
    return st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        display_toolbar=False,
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_keys[key_index],
    )


def main():
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
    if "canvas_keys" not in st.session_state:
        st.session_state.canvas_keys = [
            random_string(),
            random_string(),
            random_string(),
            random_string(),
        ]

    st.title("Digit Recogniser")
    st.info(INTRO)

    # Add custom CSS to remove gaps between columns
    st.html(
        """
        <style>
        .stColumn {
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        .stMainBlockContainer {
            max-width:590px;
        }
        .stAlert {
          margin-bottom: 2rem;
        }
        
        .stHorizontalBlock .stVerticalBlock {
            justify-content: flex-start;
        }
        .stHorizontalBlock .stElementContainer {
          margin-top: -2rem;
        }
        
        .block-container {
            padding-top: 1.5rem;
        }
        /* Green button for Predict */
        .st-key-predict_btn > .stButton button {
            background-color: #3CB371 !important;
            color: white !important;
            margin-top: 0;
        }
        /* Red button for Clear */
        .st-key-clear_btn > .stButton button {
            background-color: #DC143C !important;
            color: white !important;
            margin-top: 0;
        }
        .prediction {
            border:3px solid #3CB371; 
            border-radius:12px; 
            text-align:center; 
            font-size:3rem;
            color:#3CB371;
            margin-top: 1rem;  
            margin-bottom: 1rem;
            font-weight:900;
        }
        </style>
        """
    )
    left, right = st.columns(2, gap=None)
    with left:
        canvasTL = make_canvas(0)
        canvasBL = make_canvas(2)
    with right:
        canvasTR = make_canvas(1)
        canvasBR = make_canvas(3)

    model = load_model()

    if st.button("Predict", type="primary", key="predict_btn"):
        tl = preprocess_image(canvasTL.image_data)
        tr = preprocess_image(canvasTR.image_data)
        bl = preprocess_image(canvasBL.image_data)
        br = preprocess_image(canvasBR.image_data)

        composite = assemble_composite(tl, tr, bl, br).to(device)

        with torch.no_grad():
            # greedy autoregressive decode – predict up to 4 digits
            input_seq = torch.full(
                (1, 5), utils.BLANK_TOKEN, device=device, dtype=torch.long
            )
            input_seq[0, 0] = utils.START_TOKEN

            output_digits = []
            for pos in range(4):  # decoder positions 0…3
                logits = model(composite, input_seq)  # (1, 5, vocab)
                next_token = logits[0, pos].argmax().item()

                if next_token == utils.END_TOKEN:
                    break

                output_digits.append(next_token)
                input_seq[0, pos + 1] = next_token  # feed predicted token back

            st.session_state.prediction = output_digits
        st.session_state.has_prediction = True

    if st.session_state.has_prediction:
        # Convert to display-friendly strings
        def pretty(d):
            return "B" if d == 12 else str(d)

        preds = [pretty(d) for d in st.session_state.prediction]
        # Pad to 4 predictions in case of early END
        while len(preds) < 4:
            preds.append("")

        # Display in a 2x2 grid: TL, TR, BL, BR
        st.markdown(
            "<h3 style='text-align: center; color: #3CB371'>Predicted digits</h4>",
            unsafe_allow_html=True,
        )
        top = st.columns(2, gap="small")
        with top[0]:
            st.markdown(
                f"<div class='prediction'>{preds[0]}</div>",
                unsafe_allow_html=True,
            )
        with top[1]:
            st.markdown(
                f"<div class='prediction'>{preds[1]}</div>",
                unsafe_allow_html=True,
            )
        bottom = st.columns(2, gap="small")
        with bottom[0]:
            st.markdown(
                f"<div class='prediction'>{preds[2]}</div>",
                unsafe_allow_html=True,
            )
        with bottom[1]:
            st.markdown(
                f"<div class='prediction'>{preds[3]}</div>",
                unsafe_allow_html=True,
            )

    if st.button("Clear All", type="secondary", key="clear_btn"):
        st.session_state.canvas_keys = [
            random_string(),
            random_string(),
            random_string(),
            random_string(),
        ]
        st.session_state.prediction = None
        st.session_state.has_prediction = False
        st.rerun()


if __name__ == "__main__":
    main()
