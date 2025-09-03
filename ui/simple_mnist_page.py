import base64
import io
import os
import random
import string

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

import db
from common import utils
from models import CNN

# ----- Config: set these in your env on the Lightsail box -----
# export RUNPOD_ENDPOINT="https://api.runpod.ai/v2/<your-endpoint>/runsync"
# export RUNPOD_API_KEY="rp_xxx..."
RUNPOD_ENDPOINT = os.environ.get("RUNPOD_ENDPOINT", "").strip()
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()

INTRO = """
This is a demonstration app showing simple handwriting recognition.

The model is trained on MNIST (PyTorch). Draw a digit (0–9) and press **Predict**.
You can then enter the true label and press **Submit** to record it server-side.
"""
MODEL = None

TEMPERATURE = 2.0
_DB_READY = False


def _ensure_db():
    global _DB_READY
    if not _DB_READY:
        db.setup_database()
        _DB_READY = True


@st.cache_resource
def load_model_once():
    global MODEL
    if MODEL is None:
        model = CNN.CNN()
        state_dict = torch.load(CNN.MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        MODEL = model
    return MODEL


@torch.no_grad()
def do_predict(png_b64) -> dict:
    model = load_model_once()
    x = preprocess_image_from_png_b64(png_b64)
    logits = model(x)
    scaled = logits / TEMPERATURE
    probs = torch.softmax(scaled, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item() * 100.0)
    return {"ok": True, "data": {"prediction": pred, "confidence": conf}}


def do_submit(pred: int, conf: float, label: int) -> dict:
    try:
        _ensure_db()
        db.log_prediction(pred, conf, label)
    except Exception as e:
        # If DB isn't configured for serverless, return an explicit error
        return {"ok": False, "error": f"failed to log prediction: {e}"}
    return {"ok": True, "data": {"logged": True}}


def do_list() -> dict:
    try:
        _ensure_db()

        df = db.get_all_predictions()  # returns a DataFrame
        if df is None or df.empty:
            rows = []
        else:
            # make sure timestamps are JSON-serializable
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            # take first N and convert to list[dict]
            rows = df.head(50).to_dict(orient="records")

            # avoid numpy types that can trip some JSON encoders
            for r in rows:
                for k, v in list(r.items()):
                    if hasattr(v, "item"):  # numpy scalar -> python scalar
                        r[k] = v.item()

        return {"ok": True, "rows": rows}
    except Exception as e:
        return {"ok": False, "error": f"failed to list predictions: {e}"}


def preprocess_image_from_png_b64(png_b64: str) -> torch.Tensor:
    raw = base64.b64decode(png_b64.encode("ascii"))
    image = Image.open(io.BytesIO(raw)).convert("L")

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        # square pad, centered, with margin
        w, h = image.size
        desired = max(w, h) + 40
        pad_left = (desired - w) // 2
        pad_top = (desired - h) // 2
        pad_right = desired - w - pad_left
        pad_bottom = desired - h - pad_top
        image = ImageOps.expand(
            image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0
        )

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Lambda(
                lambda img: img.point(lambda p: 255 if p > 50 else 0, "L")
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)  # [1, 1, 28, 28]


def _rand():
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )


def image_to_png_b64(image_array) -> str:
    """Canvas -> PNG base64 (so we keep payloads small and lossless)."""
    pil = Image.fromarray(image_array).convert("L")  # grayscale is fine
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    utils.setup_logging()
    st.title("Digit Recogniser")
    st.markdown(INTRO)

    # session
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = _rand()

    # canvas
    canvas = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("Predict", type="primary", disabled=canvas.image_data is None):
            if canvas.image_data is None:
                st.warning("Draw a digit first.")
            else:
                png_b64 = image_to_png_b64(canvas.image_data)
                res = do_predict(png_b64)
                if not res.get("ok"):
                    st.error(res.get("error", "Unknown error"))
                else:
                    data = res["data"]
                    st.session_state.prediction = data["prediction"]
                    st.session_state.confidence = data["confidence"]
                    st.session_state.has_prediction = True

    if st.session_state.has_prediction:
        st.write(f"**Prediction:** {st.session_state.prediction}")
        st.write(f"**Confidence:** {st.session_state.confidence:.1f}%")

        true_label = st.number_input(
            "**True label:**", min_value=0, max_value=9, step=1, value=None
        )

        with cols[1]:
            if st.button("Submit", type="secondary", disabled=true_label is None):
                res = do_submit(
                    int(st.session_state.prediction),
                    float(st.session_state.confidence),
                    int(true_label),
                )
                if not res.get("ok"):
                    st.error(res.get("error", "Submission failed"))
                else:
                    st.session_state.has_prediction = False
                    st.session_state.canvas_key = _rand()
                    st.success("Submission recorded.")
                    st.rerun()
    # Always show all submissions
    res_list = do_list()
    if res_list.get("ok") and "rows" in res_list:
        st.subheader("All Submissions")
        st.dataframe(pd.DataFrame(res_list["rows"]))
    elif not res_list.get("ok"):
        st.error(
            f"Failed to fetch submissions: {res_list.get('error', 'Unknown error')}"
        )


if __name__ == "__main__":
    main()
