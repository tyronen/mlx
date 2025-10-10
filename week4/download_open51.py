import fiftyone
import utils
import time
import logging
from pathlib import Path


def main():
    utils.setup_logging()

    # Start timing
    start_time = time.time()

    logging.info("Starting Open Images v7 download (JPEGs only)...")
    logging.info("Split: validation, Max samples: 100")

    # Track dataset loading time
    load_start = time.time()
    dataset = fiftyone.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=200,
        only_matching=True,
        classes=None,  # Download all classes but only images
        attrs=None,  # Skip attribute annotations
        label_types=None,  # Skip all label types (no CSVs)
    )
    load_time = time.time() - load_start

    # Get dataset statistics
    total_samples = len(dataset)

    # Calculate file sizes and statistics
    total_size = 0
    file_count = 0

    # Get file paths from the dataset samples
    for sample in dataset:
        if sample.filepath and Path(sample.filepath).exists():
            file_path = Path(sample.filepath)
            if file_path.suffix.lower() in [".jpg", ".jpeg"]:
                total_size += file_path.stat().st_size
                file_count += 1

    total_time = time.time() - start_time
    total_mb = total_size / (1024 * 1024)

    # Performance analysis
    logging.info(f"\n=== Performance Analysis ===")
    logging.info(f"Dataset loading time: {load_time:.1f}s")
    logging.info(f"Total download time: {total_time:.1f}s")
    logging.info(f"Images downloaded: {file_count}")
    logging.info(f"Total data: {total_mb:.1f}MB")

    if total_time > 0:
        throughput = total_mb / total_time
        images_per_sec = file_count / total_time
        logging.info(f"Throughput: {throughput:.2f}MB/s")
        logging.info(f"Network utilization: {(throughput * 8):.1f}Mbps")
        logging.info(f"Images per second: {images_per_sec:.1f}")

    if file_count > 0:
        avg_size = (total_size / file_count) / 1024  # KB
        logging.info(f"Average image size: {avg_size:.1f}KB")

    logging.info(f"Total samples in dataset: {total_samples}")


if __name__ == "__main__":
    main()
