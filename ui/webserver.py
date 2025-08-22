import base64
import io
import os
import random
import string

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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


def call_runpod(action: str, payload: dict) -> dict:
    """POST to RunPod Serverless with a minimal input schema."""
    if not RUNPOD_ENDPOINT or not RUNPOD_API_KEY:
        return {"ok": False, "error": "RUNPOD_ENDPOINT / RUNPOD_API_KEY not set"}
    r = requests.post(
        RUNPOD_ENDPOINT,
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"input": {"action": action, "payload": payload}},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    # RunPod returns output under 'output' for runsync; normalize shape
    return data.get("output", data)


def main():
    st.set_page_config(page_title="Digit Recogniser", page_icon="🔢", layout="centered")
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
                res = call_runpod("predict", {"png_b64": png_b64})
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
                res = call_runpod(
                    "submit",
                    {
                        "prediction": int(st.session_state.prediction),
                        "confidence": float(st.session_state.confidence),
                        "true_label": int(true_label),
                    },
                )
                if not res.get("ok"):
                    st.error(res.get("error", "Submission failed"))
                else:
                    st.session_state.has_prediction = False
                    st.session_state.canvas_key = _rand()
                    st.success("Submission recorded.")
                    st.rerun()

    st.caption(
        "⚙️ UI runs on CPU here; inference & logging happen on RunPod Serverless."
    )


if __name__ == "__main__":
    main()
