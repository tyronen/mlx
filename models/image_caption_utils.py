import kagglehub
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import csv
import os

BASE_FLICKR_MODEL_FILE = "data/base_flickr_model.pth"
BASE_COCO_MODEL_FILE = "data/base_coco_model.pth"
CUSTOM_FLICKR_MODEL_FILE = "data/custom_flickr_model.pth"
CUSTOM_COCO_MODEL_FILE = "data/custom_coco_model.pth"
DATA_FRACTION = 0.004

TOKENIZER = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
)
TOKENIZER.add_special_tokens({"additional_special_tokens": ["<IMG>"]})
TOKENIZER.pad_token = TOKENIZER.eos_token
# Set BOS token to EOS token if it doesn't exist
if TOKENIZER.bos_token is None:
    TOKENIZER.bos_token = TOKENIZER.eos_token


def get_flickr(test_mode=False):
    imagepath = kagglehub.dataset_download("adityajn105/flickr30k")
    image_dir = f"{imagepath}/Images"
    filenames, rows = get_images_and_captions(
        f"{imagepath}/captions.txt", "image", image_dir, test_mode
    )
    return image_dir, filenames, rows


def get_coco(test_mode=False):
    image_dir = "data/coco"
    filenames, rows = get_images_and_captions(
        "data/coco/captions.csv", "file_name", image_dir, test_mode
    )
    return image_dir, filenames, rows


def get_images_and_captions(captions_path, field_name, image_dir, test_mode=False):
    with open(captions_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        rows.sort(
            key=lambda r: r[field_name]
        )  # deterministic ordering by image filename

    if test_mode and DATA_FRACTION < 1:
        num_rows = max(1, int(DATA_FRACTION * len(rows)))
        rows = rows[:num_rows]

    # Get unique filenames and check which ones exist
    unique_filenames = list({row[field_name] for row in rows})
    existing_filenames = set()
    missing_count = 0

    for filename in unique_filenames:
        filepath = os.path.join(image_dir, filename)
        if os.path.exists(filepath):
            existing_filenames.add(filename)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} image files not found in {image_dir}")

    # Filter rows to only include existing images
    filtered_rows = [row for row in rows if row[field_name] in existing_filenames]
    filenames = [f for f in unique_filenames if f in existing_filenames]

    print(f"Loaded {len(filenames)} images with {len(filtered_rows)} captions")

    return filenames, filtered_rows


def collate_fn(batch):
    images, input_ids = zip(*batch)
    images = torch.stack(images)  # [B, 768]
    input_ids = torch.stack(input_ids)  # [B, L]
    return {"images": images, "input_ids": input_ids}


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, device, batch_size, train=False):
        # Reduced workers to avoid "Too many open files" with large batch sizes
        num_workers = 4 if device.type == "cuda" else 0 if device.type == "mps" else 2
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            prefetch_factor=8,  # Increased prefetch for better GPU saturation
            collate_fn=collate_fn,
        )
