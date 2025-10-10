#!/usr/bin/env python3
"""
Download Open Images V7 dataset - much faster and higher quality than Wikimedia Commons.
No rate limiting, hosted on Google Cloud Storage.
"""
# Download metadata
import requests
import asyncio
import httpx
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
import os
import utils

# Suppress httpx info logging for successful requests
logging.getLogger("httpx").setLevel(logging.WARNING)

USER_AGENT = (
    "gg-clip-vit/0.1 (https://github.com/tyrone/gg-clip-vit; tyrone.nicholas@gmail.com)"
)


def get_open_images_urls(num_images):
    """Get Open Images URLs - much faster than Wikimedia API calls."""
    logging.info("Downloading Open Images metadata...")

    # Use the correct metadata URL from the official repository
    # This contains the list of train image IDs
    metadata_url = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"

    try:
        response = requests.get(metadata_url, timeout=60)
        response.raise_for_status()
        logging.info(f"Downloaded {len(response.content)} bytes of metadata")
    except Exception as e:
        logging.info(f"Failed to download metadata: {e}")
        return []

    # Save temporarily
    metadata_path = Path("open_images_metadata.csv")
    metadata_path.write_bytes(response.content)

    try:
        # Read with pandas for easy filtering
        df = pd.read_csv(metadata_path)
        logging.info(f"Loaded metadata with {len(df)} images")

        # Sample the requested number
        df_sample = df.sample(n=min(num_images, len(df)), random_state=42)

        images = []
        for _, row in df_sample.iterrows():
            image_id = row["ImageID"]
            # Open Images URLs are predictable and fast
            url = f"https://storage.googleapis.com/openimages/v7/train/{image_id}.jpg"
            filename = f"{image_id}.jpg"

            images.append({"filename": filename, "url": url, "image_id": image_id})

    except Exception as e:
        logging.info(f"Error processing metadata: {e}")
        images = []
    finally:
        # Cleanup
        if metadata_path.exists():
            metadata_path.unlink()

    logging.info(f"Prepared {len(images)} Open Images URLs")
    return images


async def download_sample(sample_images, max_connections=100):
    """Download all images concurrently - Open Images has no rate limiting!"""
    total = len(sample_images)
    timeout = httpx.Timeout(30.0, connect=10.0)

    semaphore = asyncio.Semaphore(max_connections)

    # Higher limits since Open Images doesn't rate limit
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
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
                success, download_time, write_time, size = await _download_one_async(
                    img, client
                )
                if success:
                    download_times.append(download_time)
                    write_times.append(write_time)
                    sizes.append(size)
                pbar.update()
                return success

        start_time = time.time()
        tasks = [asyncio.create_task(_worker(img)) for img in sample_images]
        results = await asyncio.gather(*tasks)
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
            logging.info(f"Images downloaded: {sum(results)}")
            logging.info(f"Average download time: {avg_download:.3f}s")
            logging.info(f"Average write time: {avg_write:.3f}s")
            logging.info(f"Average image size: {avg_size:.1f}KB")
            logging.info(f"Total data: {total_mb:.1f}MB")
            logging.info(f"Throughput: {throughput:.2f}MB/s")
            logging.info(
                f"Network utilization: {(throughput * 8):.1f}Mbps of your 45Mbps"
            )


async def _download_one_async(image_dict, client):
    """Async downloader - Open Images should be much more reliable."""
    start_time = time.time()
    out_path = Path(f"data/open_images/{image_dict['filename']}")
    if out_path.exists():
        return 1, 0, 0, 0  # success, download_time, write_time, size

    for attempt in range(3):
        try:
            download_start = time.time()
            r = await client.get(image_dict["url"])
            download_time = time.time() - download_start

            if r.status_code == 200:
                write_start = time.time()
                out_path.write_bytes(r.content)
                write_time = time.time() - write_start
                total_time = time.time() - start_time
                size = len(r.content)
                return 1, download_time, write_time, size
            else:
                logging.error(
                    f"{r.status_code} {r.reason_phrase} {image_dict['filename']} (try {attempt+1})"
                )
        except httpx.HTTPError as exc:
            logging.error(f"HTTP error {exc} on {image_dict['url']} (try {attempt+1})")

        await asyncio.sleep(2**attempt)  # Exponential backoff
    return 0, 0, 0, 0


def main():
    utils.setup_logging()
    os.makedirs("data/open_images", exist_ok=True)

    # Test with smaller batch first
    sample_images = get_open_images_urls(100)  # Change to 60000 for full dataset
    asyncio.run(download_sample(sample_images))


if __name__ == "__main__":
    main()
