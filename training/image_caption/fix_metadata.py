#!/usr/bin/env python3
"""
Fix metadata.csv captions that start with "of" and apply proper sentence case.

This script:
1. Reads metadata.csv
2. For each caption that starts with "of ", removes the "of " prefix
3. Applies sentence case (capitalize first letter)
4. Writes the fixed captions back to metadata.csv (creating a backup first)
"""

import csv
import pathlib
import logging
from common import utils


def fix_caption(text: str) -> str:
    """
    Fix a caption by removing leading 'of' and applying sentence case.

    Args:
        text: The caption text to fix

    Returns:
        The fixed caption text
    """
    text = text.strip()

    # Remove leading "of " (case insensitive)
    if text.lower().startswith("of "):
        text = text[3:]  # Remove "of "

    # Apply sentence case: capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text


def main():
    utils.setup_logging()

    input_path = pathlib.Path("data/coco/metadata.csv")
    output_path = pathlib.Path("data/coco/captions.csv")

    if not input_path.exists():
        logging.error(f"Error: {input_path} does not exist")
        return

    # Read and fix the data
    fixed_data = []
    with input_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read header
        fixed_data.append(header)

        fixed_count = 0
        total_count = 0

        for row in reader:
            if len(row) == 2:
                file_name, text = row
                original_text = text
                fixed_text = fix_caption(text)

                if original_text != fixed_text:
                    fixed_count += 1
                    if fixed_count <= 10:  # Show first 10 examples
                        logging.info(f"Fixed: '{original_text}' -> '{fixed_text}'")

                fixed_data.append([file_name, fixed_text])
                total_count += 1

    # Write fixed data back
    logging.info(f"\nWriting fixed data to {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(fixed_data)

    logging.info(f"\nSummary:")
    logging.info(f"  Total captions: {total_count}")
    logging.info(f"  Fixed captions: {fixed_count}")
    logging.info(f"  Unchanged captions: {total_count - fixed_count}")
    logging.info(f"\New data saved to: {output_path}")


if __name__ == "__main__":
    main()
