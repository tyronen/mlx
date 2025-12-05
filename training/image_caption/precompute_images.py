import json
import os
import math
import shutil
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from common import arguments
from models import image_caption, image_caption_utils


class ImageFeatureDataset(Dataset):
    def __init__(self, image_filenames, image_dir):
        self.image_filenames = image_filenames
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")
        return filename, image


def collate_fn(batch):
    filenames, images = zip(*batch)
    return list(filenames), list(images)


def human_readable_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} EB"


def ensure_disk_space(output_path: str, num_images: int, tokens_per_image: int):
    # CLIP vision hidden dim is 1024 and we store bf16 (~2 bytes) tensors.
    hidden_dim = 1024
    dtype_bytes = 2
    safety_factor = 1.1
    required = num_images * tokens_per_image * hidden_dim * dtype_bytes
    required = math.ceil(required * safety_factor)
    target_dir = os.path.dirname(output_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    if free_bytes < required:
        raise RuntimeError(
            (
                "Not enough free space to write precomputed features. "
                f"Required ≈ {human_readable_bytes(required)}, "
                f"available {human_readable_bytes(free_bytes)}."
            )
        )
    print(
        f"Estimated storage needed: {human_readable_bytes(required)} "
        f"for {num_images} images ({tokens_per_image} tokens each)."
    )


def write_metadata(meta_path: str, metadata: dict):
    torch.save(metadata, meta_path)
    json_path = meta_path.replace(".pt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(dataset_name, test_mode, official_captions, max_vision_tokens):
    model = image_caption.ImageEncoder(max_vision_tokens=max_vision_tokens)
    model.eval()

    if dataset_name == "flickr":
        imagepath, image_filenames, _ = image_caption_utils.get_flickr(test_mode)
    elif dataset_name == "coco":
        imagepath, image_filenames, _ = image_caption_utils.get_coco(
            test_mode, official_captions
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'coco' or 'flickr'.")

    metadata_path, feature_path = image_caption_utils.feature_paths(
        dataset_name,
        official_captions if dataset_name == "coco" else False,
        max_vision_tokens,
    )
    output_dir = os.path.dirname(metadata_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Estimate storage before allocating work
    total_images = len(image_filenames)
    target_tokens = 257
    if max_vision_tokens is not None and max_vision_tokens > 0:
        target_tokens = min(max_vision_tokens, target_tokens)
    ensure_disk_space(metadata_path, total_images, target_tokens)

    # Dataset and DataLoader
    dataset = ImageFeatureDataset(image_filenames, imagepath)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    feature_shape = (total_images, target_tokens, 1024)
    feature_dtype = np.float16
    if os.path.exists(feature_path):
        os.remove(feature_path)
    features_memmap = np.memmap(
        feature_path,
        mode="w+",
        dtype=feature_dtype,
        shape=feature_shape,
    )
    ordered_filenames: List[str] = []
    write_idx = 0

    with torch.no_grad():
        for filenames, images in tqdm(dataloader):
            pixel_values = model.processor(
                images=images, return_tensors="pt"
            ).pixel_values
            outputs = model(pixel_values).to(torch.float16).cpu()
            batch_size = outputs.size(0)
            if outputs.size(1) != target_tokens:
                raise ValueError(
                    f"Expected {target_tokens} vision tokens, got {outputs.size(1)}"
                )
            features_memmap[write_idx : write_idx + batch_size] = outputs.numpy()
            write_idx += batch_size
            ordered_filenames.extend(filenames)

    features_memmap.flush()
    del features_memmap

    if write_idx != total_images:
        raise RuntimeError(
            f"Wrote {write_idx} embeddings but expected {total_images}. "
            "Check dataset splits for duplicates or missing images."
        )

    metadata = {
        "feature_path": os.path.basename(feature_path),
        "shape": feature_shape,
        "dtype": str(feature_dtype),
        "filenames": ordered_filenames,
        "dataset": dataset_name,
        "use_official_captions": official_captions if dataset_name == "coco" else False,
        "max_vision_tokens": target_tokens,
    }
    write_metadata(metadata_path, metadata)
    print(
        f"Saved {total_images} image embeddings to {feature_path} with metadata {metadata_path}"
    )


if __name__ == "__main__":
    parser = arguments.get_parser(
        description="Precompute image embeddings for image captioning"
    )
    args = parser.parse_args()
    main(
        args.dataset,
        args.check,
        args.official_captions,
        args.max_vision_tokens,
    )
