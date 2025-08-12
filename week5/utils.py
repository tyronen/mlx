import logging
from contextlib import nullcontext
import kagglehub
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import csv

BASE_MODEL_FILE = "data/base_model.pth"
CUSTOM_MODEL_FILE = "data/custom_model.pth"
DATA_FRACTION = 0.004

TOKENIZER = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
)
TOKENIZER.add_special_tokens({"additional_special_tokens": ["<IMG>"]})
TOKENIZER.pad_token = TOKENIZER.eos_token
# Set BOS token to EOS token if it doesn't exist
if TOKENIZER.bos_token is None:
    TOKENIZER.bos_token = TOKENIZER.eos_token


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
        return autocast(), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)


def get_captions():
    imagepath = kagglehub.dataset_download("adityajn105/flickr30k")

    # Load all caption rows as full dictionaries (keep every field)
    with open(f"{imagepath}/captions.txt", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: r["image"])  # deterministic ordering by image filename

    if DATA_FRACTION < 1:
        num_rows = max(1, int(DATA_FRACTION * len(rows)))
        rows = rows[:num_rows]

    filenames = list({row["image"] for row in rows})

    return imagepath, filenames, rows


def collate_fn(batch):
    images, input_ids = zip(*batch)
    images = torch.stack(images)  # [B, 768]
    input_ids = torch.stack(input_ids)  # [B, L]
    return {"images": images, "input_ids": input_ids}


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, device, batch_size, train=False):
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 4
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )
