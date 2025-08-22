import base64
import io

import runpod
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from common import arguments
from db import log_prediction, setup_database
from models import CNN

TEMPERATURE = 2.0

# --- Lazy model cache (pod stays warm; model loads once) ---
MODEL = None


args = arguments.get_args("CNN API server")

_DB_READY = False


def _ensure_db():
    global _DB_READY
    if not _DB_READY:
        setup_database()
        _DB_READY = True


def load_model_once():
    global MODEL
    if MODEL is None:
        model = CNN.CNN()
        state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        MODEL = model
    return MODEL


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


# --- Actions ---
@torch.no_grad()
def do_predict(payload: dict) -> dict:
    png_b64 = payload.get("png_b64")
    if not png_b64:
        return {"ok": False, "error": "missing 'png_b64' in payload"}
    model = load_model_once()
    x = preprocess_image_from_png_b64(png_b64)
    logits = model(x)
    scaled = logits / TEMPERATURE
    probs = torch.softmax(scaled, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item() * 100.0)
    return {"ok": True, "data": {"prediction": pred, "confidence": conf}}


def do_submit(payload: dict) -> dict:
    try:
        pred = int(payload["prediction"])
        conf = float(payload["confidence"])
        true = int(payload["true_label"])
    except Exception:
        return {
            "ok": False,
            "error": "payload must include prediction, confidence, true_label",
        }

    try:
        _ensure_db()
        log_prediction(pred, conf, true)
    except Exception as e:
        # If DB isn't configured for serverless, return an explicit error
        return {"ok": False, "error": f"failed to log prediction: {e}"}
    return {"ok": True, "data": {"logged": True}}


# --- Router & handler ---
ACTIONS = {
    "predict": do_predict,
    "submit": do_submit,
}


def handler(event):
    """
    Expected input:
    {
      "input": {
        "action": "predict" | "submit",
        "payload": { ... }   # per action
      }
    }
    """
    try:
        inp = event.get("input") or {}
        action = inp.get("action")
        payload = inp.get("payload") or {}
        fn = ACTIONS.get(action)
        if not fn:
            return {
                "ok": False,
                "error": f"unknown action '{action}'",
                "actions": list(ACTIONS.keys()),
            }
        return fn(payload)
    except Exception as e:
        return {"ok": False, "error": f"unhandled server error: {e}"}


runpod.serverless.start({"handler": handler})
