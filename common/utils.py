import logging
import random
import re
import unicodedata
from contextlib import nullcontext

import numpy as np
import torch
from torch.amp import autocast, GradScaler


def randomize():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return autocast(device_type="cuda"), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)


TOKEN_RE = re.compile(r"[A-Za-z0-9.+#_]+")


def normalize_text(s: str) -> str:
    # casefold + unicode normalize
    return unicodedata.normalize("NFKC", s).casefold()


def tokenize_text(s: str) -> list[str]:
    s = normalize_text(s or "")
    return TOKEN_RE.findall(s)
