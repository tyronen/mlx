from pathlib import Path

from PIL import Image
import requests
import time
import os
import logging
import threading
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import asyncio
import httpx
import time

import utils
from tqdm import tqdm
import logging

# Suppress httpx info logging for successful requests
logging.getLogger("httpx").setLevel(logging.WARNING)

USER_AGENT = (
    "gg-clip-vit/0.1 (https://github.com/tyrone/gg-clip-vit; tyrone.nicholas@gmail.com)"
)

_thread_local = threading.local()


def get_random_commons_images(num_images):
    images = []
    pbar = tqdm(total=num_images, desc="Collecting images")

    filenames = set()
    while len(images) < num_images:
        # NO API KEY NEEDED!
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            headers={"User-Agent": USER_AGENT},
            params={
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": 6,  # File namespace
                "grnlimit": 50,  # Max per request
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
            },
        )

        data = response.json()
        if "query" in data and "pages" in data["query"]:
            batch_count = 0
            for page in data["query"]["pages"].values():
                filename = page["title"].replace("File:", "")
                if filename in filenames:
                    continue
                if "imageinfo" in page:
                    info = page["imageinfo"][0]
                    if info["mime"] not in ("image/jpeg", "image/png"):
                        continue
                    if info.get("width", 0) >= 224 and info.get("height", 0) >= 224:
                        # Convert to proper thumbnail URL using Wikimedia's thumb service
                        original_url = info["url"]
                        if (
                            "upload.wikimedia.org/wikipedia/commons/" in original_url
                            and "/thumb/" not in original_url
                        ):
                            # Convert: .../commons/a/ab/file.jpg -> .../commons/thumb/a/ab/file.jpg/640px-file.jpg
                            parts = original_url.split("/wikipedia/commons/")
                            if len(parts) == 2:
                                base_url = parts[0]
                                file_path = parts[1]
                                thumb_url = f"{base_url}/wikipedia/commons/thumb/{file_path}/640px-{filename}"
                            else:
                                thumb_url = original_url
                        else:
                            thumb_url = original_url

                        images.append(
                            {
                                "filename": filename,
                                "url": thumb_url,
                            }
                        )
                        filenames.add(filename)
                        batch_count += 1

            pbar.update(batch_count)
            pbar.set_postfix({"total": len(images)})
        time.sleep(0.1)  # Be nice to their servers
    pbar.close()
    return images[:num_images]


async def download_sample(sample_images, max_connections=60):
    """Download all images concurrently using HTTP/2 with httpx.AsyncClient."""
    total = len(sample_images)
    timeout = httpx.Timeout(20.0, connect=8.0)  # Balanced timeouts

    semaphore = asyncio.Semaphore(max_connections)

    # Configure client with balanced connection limits
    limits = httpx.Limits(max_keepalive_connections=30, max_connections=80)
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
                success, download_time, write_time, size = await _download_one_async(img, client)
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

        # Print profiling results
        if download_times:
            avg_download = sum(download_times) / len(download_times)
            avg_write = sum(write_times) / len(write_times)
            avg_size = sum(sizes) / len(sizes) / 1024  # KB
            total_mb = sum(sizes) / (1024 * 1024)
            throughput = total_mb / total_time

            print(f"\n=== Performance Analysis ===")
            print(f"Total time: {total_time:.1f}s")
            print(f"Images downloaded: {sum(results)}")
            print(f"Average download time: {avg_download:.3f}s")
            print(f"Average write time: {avg_write:.3f}s")
            print(f"Average image size: {avg_size:.1f}KB")
            print(f"Total data: {total_mb:.1f}MB")
            print(f"Throughput: {throughput:.2f}MB/s")
            print(f"Network utilization: {(throughput * 8):.1f}Mbps of your 45Mbps")


async def _download_one_async(image_dict, client):
    """Async downloader with retries and file‑exists skip."""
    start_time = time.time()
    out_path = Path(f"data/random_commons/{image_dict['filename']}")
    if out_path.exists():
        return 1, 0, 0, 0  # success, download_time, write_time, size

    backoff = 1
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
            elif r.status_code == 429:  # Rate limited
                # For rate limiting, use longer backoff
                backoff = min(backoff * 2, 10)  # Cap at 10 seconds
                logging.warning(
                    f"{r.status_code} {r.reason_phrase} {image_dict['filename']} (try {attempt+1}) - backing off {backoff}s"
                )
            else:
                logging.error(
                    f"{r.status_code} {r.reason_phrase} {image_dict['filename']} (try {attempt+1})"
                )
        except httpx.HTTPError as exc:
            logging.error(
                f"HTTP error {exc} on {image_dict['filename']} (try {attempt+1})"
            )

        await asyncio.sleep(backoff)
        backoff *= 2
    return 0, 0, 0, 0


def filter_quality():
    quality_images = []
    for img_path in Path("data/random_commons").glob("*"):
        try:
            img = Image.open(img_path)
            # Filter criteria
            if (
                img.size[0] >= 224
                and img.size[1] >= 224
                and img.mode in ["RGB", "L"]
                and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ):
                quality_images.append(img_path)
        except:
            continue

    print(f"Quality images: {len(quality_images)}")


def main():
    utils.setup_logging()
    os.makedirs("data/random_commons", exist_ok=True)
    sample_images = get_random_commons_images(100)
    asyncio.run(download_sample(sample_images))


if __name__ == "__main__":
    main()
