#!/usr/bin/env python3
"""
Download Open Images using TensorFlow Datasets - much cleaner approach.
"""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import utils
import logging
from PIL import Image


def download_open_images_sample(num_images, split="train"):
    """Download only the first N samples of Open Images using TFDS.
    
    This avoids downloading the entire massive dataset by using TFDS slice notation
    to only download the first num_images samples.
    """

    # Setup output directory
    output_dir = Path(f"data/open_images_tfds")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading first {num_images:,} samples from Open Images {split} split...")

    try:
        # Use slice notation to only download the first N samples
        # This should avoid downloading the entire dataset
        dataset, info = tfds.load(
            "open_images_v4",
            split=f"{split}[:{num_images}]",  # Only download first N samples
            shuffle_files=False,  # Don't shuffle to ensure we get first N
            as_supervised=False,
            with_info=True,
            download=True,  # Explicitly enable download
        )

        logging.info(f"Dataset info: {info}")
        logging.info(f"Features: {info.features}")

        # Download and save images
        logging.info(f"Downloading {num_images} images...")

        saved_count = 0
        for i, example in enumerate(
            tqdm(dataset, total=num_images, desc="Saving images")
        ):
            try:
                # Extract image and metadata
                image = example["image"].numpy()
                image_id = example.get("image/filename", f"image_{i:06d}").numpy()

                if isinstance(image_id, bytes):
                    image_id = image_id.decode("utf-8")

                # Save as PNG (preserves quality)
                output_path = output_dir / f"{image_id}.png"

                # Convert to PIL and save
                pil_image = Image.fromarray(image)
                pil_image.save(output_path)

                saved_count += 1

            except Exception as e:
                logging.warning(f"Failed to save image {i}: {e}")
                continue

        logging.info(f"Successfully saved {saved_count} images to {output_dir}")

    except Exception as e:
        logging.error(f"Failed to download Open Images: {e}")
        logging.info(
            "You may need to install tensorflow-datasets: pip install tensorflow-datasets"
        )
        return False

    return True


def main():
    utils.setup_logging()

    # Start with a small sample
    success = download_open_images_sample(num_images=100)

    if success:
        logging.info("Open Images download completed successfully!")
    else:
        logging.info("Download failed - check dependencies")


if __name__ == "__main__":
    main()
