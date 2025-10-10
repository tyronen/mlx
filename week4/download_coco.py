#!/usr/bin/env python3
"""
Download COCO dataset - much faster and higher quality than Wikimedia Commons.
No rate limiting, hosted on Microsoft Azure CDN.
"""

import asyncio
import httpx
import time
from pathlib import Path
import utils
from tqdm import tqdm
import logging
import requests
import os
import random
import zipfile
import tempfile
import json

# Suppress httpx info logging for successful requests
logging.getLogger("httpx").setLevel(logging.WARNING)

USER_AGENT = (
    "gg-clip-vit/0.1 (https://github.com/tyrone/gg-clip-vit; tyrone.nicholas@gmail.com)"
)


def get_coco_urls(num_images, split="train2017"):
    """Get COCO image URLs - much faster than Wikimedia API calls."""
    logging.info(f"Downloading COCO {split} annotations...")

    # Download COCO annotations zip file
    annotations_url = (
        f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )

    try:
        # Check if we already have the annotations file locally
        local_annotations_path = Path(f"data/annotations/instances_{split}.json")
        if local_annotations_path.exists():
            logging.info(f"Using cached annotations from {local_annotations_path}")
            with open(local_annotations_path, "r") as f:
                annotations = json.load(f)
        else:
            logging.info("Downloading COCO annotations zip file...")
            response = requests.get(annotations_url, stream=True, timeout=300)
            total_size = int(response.headers.get("content-length", 0))
            response.raise_for_status()

            # Create annotations directory
            os.makedirs("data/annotations", exist_ok=True)

            # Download and extract the zip file with progress bar
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                progress = tqdm(
                    total=total_size, unit="iB", unit_scale=True, desc="Annotations"
                )
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                    progress.update(len(chunk))
                progress.close()
                tmp_file.flush()

                logging.info("Extracting annotations...")
                with zipfile.ZipFile(tmp_file.name, "r") as zip_ref:
                    # Extract the specific instances file we need
                    instances_filename = f"annotations/instances_{split}.json"
                    if instances_filename in zip_ref.namelist():
                        zip_ref.extract(instances_filename, "data/")

                        # Load the extracted JSON
                        with open(local_annotations_path, "r") as f:
                            annotations = json.load(f)
                    else:
                        raise FileNotFoundError(
                            f"Could not find {instances_filename} in zip file"
                        )

                # Clean up temp file
                os.unlink(tmp_file.name)

        # Parse JSON
        logging.info("Parsing annotations...")

        # Extract image info
        images_info = annotations["images"]
        logging.info(f"Found {len(images_info)} images in COCO {split}")

        # Sample the requested number
        random.seed(42)  # Reproducible
        sampled_images = random.sample(images_info, min(num_images, len(images_info)))

        images = []
        for img_info in sampled_images:
            image_id = img_info["id"]
            filename = img_info["file_name"]
            # COCO URLs are predictable and fast
            url = f"http://images.cocodataset.org/{split}/{filename}"

            images.append(
                {
                    "filename": filename,
                    "url": url,
                    "image_id": image_id,
                    "width": img_info.get("width", 0),
                    "height": img_info.get("height", 0),
                }
            )

        logging.info(f"Prepared {len(images)} COCO URLs")
        return images

    except Exception as e:
        logging.error(f"Failed to download COCO metadata: {e}")
        return []


async def download_sample(sample_images, max_connections=50):
    """Download all images concurrently with conservative rate limiting."""
    total = len(sample_images)
    timeout = httpx.Timeout(30.0, connect=10.0)  # More generous timeouts

    semaphore = asyncio.Semaphore(max_connections)
    # Rate limiter: limit requests per second to avoid triggering server throttling
    rate_limiter = asyncio.Semaphore(20)  # Max 20 requests at once

    # Conservative connection limits
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=50)
    
    async with httpx.AsyncClient(
        http2=True,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
    ) as client:
        pbar = tqdm(total=total, desc="Downloading")

        # Profiling data collection
        download_times = []
        write_times = []
        sizes = []

        async def _worker(img):
            async with semaphore:
                async with rate_limiter:
                    # Small delay to spread out requests
                    await asyncio.sleep(0.05)  # 50ms delay between requests
                    success, download_time, write_time, size = await _download_one_async(
                        img, client
                    )
                    if success and size > 0:
                        download_times.append(download_time)
                        write_times.append(write_time)
                        sizes.append(size)
                    pbar.update()
                    return success

        start_time = time.time()
        tasks = [asyncio.create_task(_worker(img)) for img in sample_images]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        pbar.close()

        # logging.info profiling results
        if download_times:
            avg_download = sum(download_times) / len(download_times)
            avg_write = sum(write_times) / len(write_times)
            avg_size = sum(sizes) / len(sizes) / 1024  # KB
            total_mb = sum(sizes) / (1024 * 1024)
            throughput = total_mb / total_time

            logging.info(f"\n=== Performance Analysis ===")
            logging.info(f"Total time: {total_time:.1f}s")
            logging.info(f"Images downloaded: {len(sizes)}")
            logging.info(f"Average download time: {avg_download:.3f}s")
            logging.info(f"Average write time: {avg_write:.3f}s")
            logging.info(f"Average image size: {avg_size:.1f}KB")
            logging.info(f"Total data: {total_mb:.1f}MB")
            logging.info(f"Throughput: {throughput:.2f}MB/s")
            logging.info(f"Network utilization: {(throughput * 8):.1f}Mbps")


async def _download_one_async(image_dict, client):
    """Async downloader - COCO should be much more reliable."""
    out_path = Path(f"data/coco/{image_dict['filename']}")
    if out_path.exists():
        return 1, 0, 0, 0  # success, download_time, write_time, size

    for attempt in range(5):  # Increased retries
        try:
            download_start = time.time()
            r = await client.get(image_dict["url"])
            download_time = time.time() - download_start

            if r.status_code == 200:
                write_start = time.time()
                out_path.write_bytes(r.content)
                write_time = time.time() - write_start
                size = len(r.content)
                return 1, download_time, write_time, size
            else:
                logging.error(
                    f"{r.status_code} {r.reason_phrase} {image_dict['filename']} (try {attempt+1})"
                )
        except httpx.HTTPError as exc:
            logging.error(
                f"HTTP error {exc} on {image_dict['filename']} (try {attempt+1})"
            )

        # More gradual backoff with jitter to avoid thundering herd
        base_delay = min(2**attempt, 30)  # Cap at 30 seconds
        jitter = random.uniform(0.5, 1.5)  # Add randomness
        await asyncio.sleep(base_delay * jitter)
    return 0, 0, 0, 0


def main():
    utils.setup_logging()
    os.makedirs("data/coco", exist_ok=True)

    sample_images = get_coco_urls(60000)
    if sample_images:
        asyncio.run(download_sample(sample_images))
    else:
        logging.info("No images to download")


if __name__ == "__main__":
    main()
