import kagglehub
import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import csv
import os
import numpy as np
from functools import lru_cache
from typing import List, Tuple

FLICKR_FEATURES_PATH = "data/flickr_features.pt"
COCO_FEATURES_PATH = "data/coco_features.pt"


def feature_paths(
    dataset: str, use_official_captions: bool, max_tokens: int
) -> Tuple[str, str]:
    if dataset == "flickr":
        base = os.path.splitext(FLICKR_FEATURES_PATH)[0]
    elif dataset == "coco":
        base = os.path.splitext(COCO_FEATURES_PATH)[0]
    else:
        raise ValueError(f"Unknown dataset '{dataset}' for feature paths")
    meta_path = f"{base}_max_vision_tokens_{max_tokens}.pt"
    bin_path = f"{base}_max_vision_tokens_{max_tokens}.bin"
    return meta_path, bin_path


BASE_FLICKR_MODEL_FILE = "data/base_flickr_model.pth"
BASE_COCO_MODEL_FILE = "data/base_coco_model.pth"
OFFICIAL_COCO_MODEL_FILE = "data/official_coco_model.pth"
DATA_FRACTION = 0.004

TOKENIZER = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
)
TOKENIZER.add_special_tokens({"additional_special_tokens": ["<IMG>"]})
added_specials = {}
if TOKENIZER.bos_token is None:
    added_specials["bos_token"] = "<|im_start|>"
if TOKENIZER.pad_token is None:
    added_specials["pad_token"] = "<|im_pad|>"
if added_specials:
    TOKENIZER.add_special_tokens(added_specials)


class PrecomputedFeatureStore:
    """
    Memory-mapped access to precomputed vision embeddings keyed by filename.
    """

    def __init__(self, metadata_path: str):
        self.metadata_path = os.path.abspath(metadata_path)
        self._load_metadata()
        self._open_memmap()

    def _load_metadata(self):
        meta = torch.load(self.metadata_path)
        self.feature_path = os.path.join(
            os.path.dirname(self.metadata_path), meta["feature_path"]
        )
        self.shape = tuple(meta["shape"])
        dtype_spec = meta.get("dtype", "float16")
        # Handle both "float16" and "<class 'numpy.float16'>" formats
        if isinstance(dtype_spec, str):
            # Extract just the dtype name if it's in the "<class 'numpy.X'>" format
            if "numpy." in dtype_spec:
                dtype_spec = dtype_spec.split("numpy.")[1].rstrip("'>\"")
            self.dtype = np.dtype(dtype_spec)
        else:
            self.dtype = dtype_spec
        self.max_vision_tokens = meta.get("max_vision_tokens", self.shape[1])
        self.filenames = list(meta["filenames"])
        self.filename_to_idx = {fname: idx for idx, fname in enumerate(self.filenames)}
        self.dataset = meta.get("dataset")
        self.use_official_captions = meta.get("use_official_captions")

    def _open_memmap(self):
        self.memmap = np.memmap(
            self.feature_path, mode="r+", dtype=self.dtype, shape=self.shape
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["memmap"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._open_memmap()

    def get(self, filename: str) -> torch.Tensor:
        idx = self.filename_to_idx.get(filename)
        if idx is None:
            raise KeyError(
                f"Precomputed features missing entry for image '{filename}'. "
                "Ensure precompute_images was run with matching dataset."
            )
        array = self.memmap[idx]
        tensor = torch.from_numpy(array)
        return tensor


@lru_cache(maxsize=1)
def get_flickr(test_mode=False):
    imagepath = kagglehub.dataset_download("adityajn105/flickr30k")
    image_dir = f"{imagepath}/Images"
    filenames, rows = get_images_and_captions(
        f"{imagepath}/captions.txt", "image", image_dir, test_mode
    )
    return image_dir, filenames, rows


@lru_cache(maxsize=2)
def get_coco(test_mode=False, use_official_captions=False):
    image_dir = "data/coco"
    if use_official_captions:
        filenames, rows = get_images_and_official_captions(
            "data/annotations/captions_train2017.json", image_dir, test_mode
        )
    else:
        filenames, rows = get_images_and_captions(
            "data/coco/captions.csv", "file_name", image_dir, test_mode
        )
    return image_dir, filenames, rows


def get_images_and_official_captions(json_path, image_dir, test_mode=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a map of image_id -> filename
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # Collect all captions, mapped to their filenames
    rows = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id in image_id_to_filename:
            rows.append(
                {"file_name": image_id_to_filename[img_id], "text": ann["caption"]}
            )

    rows.sort(key=lambda r: r["file_name"])  # deterministic ordering

    if test_mode and DATA_FRACTION < 1:
        num_rows = max(1, int(DATA_FRACTION * len(rows)))
        rows = rows[:num_rows]

    # Same filtering logic as the CSV version
    unique_filenames = list({row["file_name"] for row in rows})

    # Optimize: Check against directory listing instead of os.path.exists for each file
    try:
        all_files = set(os.listdir(image_dir))
    except FileNotFoundError:
        print(f"Warning: Image directory {image_dir} not found.")
        all_files = set()

    existing_filenames = set()
    missing_count = 0

    for filename in unique_filenames:
        if filename in all_files:
            existing_filenames.add(filename)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} image files not found in {image_dir}")

    filtered_rows = [row for row in rows if row["file_name"] in existing_filenames]
    filenames = [f for f in unique_filenames if f in existing_filenames]

    print(f"Loaded {len(filenames)} images with {len(filtered_rows)} official captions")

    return filenames, filtered_rows


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

    # Optimize: Check against directory listing instead of os.path.exists for each file
    try:
        all_files = set(os.listdir(image_dir))
    except FileNotFoundError:
        print(f"Warning: Image directory {image_dir} not found.")
        all_files = set()

    existing_filenames = set()
    missing_count = 0

    for filename in unique_filenames:
        if filename in all_files:
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
    images, inputs = zip(*batch)
    images = torch.stack(images)  # [B, 768]

    # Check if inputs are lists of tensors (grouped mode) or single tensors
    if (
        isinstance(inputs[0], torch.Tensor) and inputs[0].dim() == 2
    ):  # Grouped: [num_caps, L]
        # We need to return a structure that indicates how many captions per image
        # Flatten input_ids: [B * num_caps_avg, L]
        num_captions_per_image = [inp.size(0) for inp in inputs]
        input_ids = torch.cat(inputs, dim=0)
        return {
            "images": images,
            "input_ids": input_ids,
            "num_captions_per_image": num_captions_per_image,
        }
    else:
        # Standard mode: inputs are [L] tensors
        input_ids = torch.stack(inputs)
        return {"images": images, "input_ids": input_ids}


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, device, batch_size, train=False):
        # Always use workers to hide IO latency, even for precomputed memmaps.
        # We rely on OS page cache or copy-on-read to handle concurrency.
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 2
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )
