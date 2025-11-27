import logging
import os
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch
from torch import version as torch_version
import pathlib
import csv
import time
from torch.utils.data import Dataset, DataLoader
from common import utils
from tqdm import tqdm

LLM = "Qwen/Qwen2.5-VL-3B-Instruct"


class CocoDataset(Dataset):
    def __init__(self, data_dir: str = "data/coco"):
        # Collect all JPEG/JPG files in the given directory
        data_path = pathlib.Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"{data_path} does not exist")

        self.image_paths = sorted([str(p) for p in data_path.glob("*.jpg")])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return image, image_path


def coco_collate(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)


def generate_captions(
    device, model, processor, batch_images, batch_paths, profile=False
):

    images = batch_images

    # Create per-image conversations (batched) for chat template
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "Describe this image precisely and concisely in a single sentence.",
                    },
                ],
            },
        ]
        for image in images
    ]

    # Apply chat template (batched)
    texts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # Move to device
    inputs = inputs.to(device)

    generation_kwargs = {
        "max_new_tokens": 90,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "use_cache": True,
        # Removed static cache - it triggers Triton compilation issues
        # "cache_implementation": "static",
        "num_beams": 1,
    }
    # With pad_token configured correctly, we expect the model to use the proper
    # end-of-turn token for chat/instruct-tuned models.
    try:
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            isinstance(im_end_id, int)
            and im_end_id != processor.tokenizer.unk_token_id
            and im_end_id is not None
        ):
            generation_kwargs["eos_token_id"] = [
                im_end_id,
                processor.tokenizer.eos_token_id,
            ]
    except Exception:
        pass

    # Actual generation timing with CUDA sync for accuracy
    gen_start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    # Sync CUDA for accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_time = time.time() - gen_start

    if profile:
        logging.info(f"  Generation breakdown:")
        logging.info(f"    Model forward: {gen_time:.3f}s")
        logging.info(
            f"    Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}"
        )
        logging.info(f"    Generated shape: {generated_ids.shape}")
        logging.info(
            f"    Generated length: {generated_ids.shape[1] - inputs['input_ids'].shape[1]} tokens"
        )

    # The issue: input has padding tokens, attention mask includes them,
    # and using attention_mask sum lands us in the padding, not at the real prompt end.
    # Solution: Find where "<|im_start|>assistant\n" ends and split there.
    captions = []

    # Get the token IDs we need to search for
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_tokens = processor.tokenizer.encode("assistant", add_special_tokens=False)
    # The newline is encoded as a specific token. We can get it by encoding a newline.
    newline_tokens = processor.tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_tokens[0] if newline_tokens else None

    if profile:
        logging.info(f"\n=== Token ID Debug ===")
        logging.info(f"im_start_id: {im_start_id}")
        logging.info(f"assistant_tokens: {assistant_tokens}")
        logging.info(f"newline_id: {newline_id}")

    # Debug: Files we're specifically interested in
    debug_files = {"000000002001.jpg", "000000005758.jpg", "000000015303.jpg"}

    for i in range(generated_ids.shape[0]):
        gen_ids = generated_ids[i].tolist()

        # Find the last occurrence of <|im_start|>assistant\n
        # This marks the start of the assistant's response
        prompt_end_pos = None

        # Search for the pattern: <|im_start|> + assistant_tokens + newline
        for pos in range(len(gen_ids) - len(assistant_tokens) - 2):
            if gen_ids[pos] == im_start_id:
                # Check if followed by assistant tokens
                matches = True
                for j, tok in enumerate(assistant_tokens):
                    if gen_ids[pos + 1 + j] != tok:
                        matches = False
                        break
                if matches and gen_ids[pos + 1 + len(assistant_tokens)] == newline_id:
                    # Found it! The response starts after the newline
                    prompt_end_pos = pos + 1 + len(assistant_tokens) + 1

        if prompt_end_pos is None:
            # Fallback: use the input length
            prompt_end_pos = inputs["input_ids"].shape[1]

        # Extract only the generated part
        generated_part = torch.tensor(
            gen_ids[prompt_end_pos:], device=generated_ids.device
        )

        # Debug specific problematic images
        filename = pathlib.Path(batch_paths[i]).name
        if filename in debug_files:
            logging.info(f"\n=== DEBUG {filename} (item {i}) ===")
            logging.info(f"Found prompt_end_pos: {prompt_end_pos}")
            logging.info(f"Generated sequence length: {len(gen_ids)}")

            # Show tokens around the boundary
            boundary_start = max(0, prompt_end_pos - 10)
            boundary_end = min(len(gen_ids), prompt_end_pos + 20)
            boundary_ids = gen_ids[boundary_start:boundary_end]
            boundary_tokens = processor.tokenizer.convert_ids_to_tokens(boundary_ids)
            logging.info(f"\nBoundary tokens [{boundary_start}:{boundary_end}]:")
            for idx, (token_id, token) in enumerate(
                zip(boundary_ids, boundary_tokens), start=boundary_start
            ):
                marker = " <-- NEW SPLIT HERE" if idx == prompt_end_pos else ""
                logging.info(f"  [{idx}] {token_id:6d} -> {token!r}{marker}")

            # Decode the generated part with and without special tokens
            logging.info(f"\nGenerated part shape: {generated_part.shape}")
            logging.info(
                f"Generated part IDs (first 30): {generated_part[:30].tolist()}"
            )

            decoded_with_special = processor.decode(
                generated_part,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            decoded_without_special = processor.decode(
                generated_part,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            logging.info(f"\nDecoded WITH special tokens: {decoded_with_special!r}")
            logging.info(f"Decoded WITHOUT special tokens: {decoded_without_special!r}")

        caption = processor.decode(
            generated_part,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        captions.append(caption.strip())

    return captions


def main():
    utils.setup_logging()

    # Diagnostic info to help debug GPU visibility issues
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        cuda_version = getattr(torch_version, "cuda", "unknown")
        logging.info(f"CUDA version: {cuda_version}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logging.info(f"  Capability: {torch.cuda.get_device_capability(i)}")
    else:
        logging.error("❌ CUDA is NOT available! Check:")
        logging.error("  1. nvidia-smi works in the container")
        logging.error(
            "  2. Container started with --gpus all (Docker) or GPU enabled (RunPod)"
        )
        logging.error("  3. PyTorch was built with CUDA support")
        logging.error("  4. NVIDIA Container Toolkit is installed on host")

    device = utils.get_device()
    logging.info(f"Using device: {device}")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load the model and processor with optimizations
    model = AutoModelForImageTextToText.from_pretrained(
        LLM,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=(
            "flash_attention_2" if torch.cuda.is_available() else "eager"
        ),
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        use_safetensors=True,  # Use safetensors for faster loading
    )
    # This is the key: control vision tokens by setting pixel limits in the processor
    # Lower resolution = fewer vision tokens = faster generation
    # These settings correspond to roughly 256x256 to 448x448 resolution.
    processor = AutoProcessor.from_pretrained(
        LLM,
        min_pixels=256 * 256,
        max_pixels=336 * 336,
    )
    # The tokenizer for Qwen2.5-VL doesn't have a pad token set by default.
    # Setting it to the EOS token is a common practice and helps stabilize
    # batch generation.
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|im_pad|>"
        )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # For decoder-only models, padding should be on the left
    processor.tokenizer.padding_side = "left"

    # Debug: Log token IDs to verify configuration
    logging.info(
        f"Pad token: {processor.tokenizer.pad_token} (ID: {processor.tokenizer.pad_token_id})"
    )
    logging.info(
        f"EOS token: {processor.tokenizer.eos_token} (ID: {processor.tokenizer.eos_token_id})"
    )

    coco_dataset = CocoDataset()

    # Use larger batch size - data shows larger batches are faster per image
    # Try to maximize GPU utilization while staying within memory limits
    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory > 20e9:  # > 20GB (RTX 5090 has 32GB)
            batch_size = 448  # Maximize VRAM usage for fastest throughput
        elif gpu_memory > 10e9:  # > 10GB
            batch_size = 32
        else:
            batch_size = 16
    else:
        batch_size = 6
    logging.info(f"Using batch size: {batch_size}")

    if device.type == "cuda":
        cpu_count = os.cpu_count() or 1
        # Reduced number of workers to prevent OOM with large batches
        num_workers = max(2, min(8, cpu_count))
    elif device.type == "mps":
        num_workers = 0
    else:
        num_workers = 3
    # Reduced prefetch factor to limit buffered images in RAM
    prefetch_factor = 2 if (device.type == "cuda" and num_workers > 0) else 2

    dataloader = DataLoader(
        coco_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=coco_collate,
    )

    # Use incremental writing to avoid holding all captions in memory and preventing data loss on crash
    output_csv = pathlib.Path("data/coco/metadata.csv")
    # Ensure directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mode = "w"
    # Check if we should resume (simple check: if file exists, append?
    # For now, let's overwrite to restart cleanly as requested, or append if we wanted resume logic.
    # Given the crash, a restart is safer unless we implement robust resume logic.)
    # We will use "w" to start fresh, but keep the file open.

    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "text"])  # header row

        for batch_idx, (batch_images, batch_paths) in enumerate(
            tqdm(dataloader, desc="Generating captions")
        ):
            try:
                # Profile first 3 batches to identify bottlenecks
                profile_this_batch = batch_idx < 3
                if profile_this_batch:
                    logging.info(f"\n=== Profiling Batch {batch_idx} ===")
                    if device.type == "cuda":
                        logging.info(
                            f"GPU Memory before: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB"
                        )

                captions = generate_captions(
                    device,
                    model,
                    processor,
                    batch_images,
                    batch_paths,
                    profile=profile_this_batch,
                )

                # Write immediately to file
                rows = []
                for image_path, caption in zip(batch_paths, captions):
                    rows.append([pathlib.Path(image_path).name, caption])
                writer.writerows(rows)
                csvfile.flush()  # Ensure data is written to disk

                if profile_this_batch and device.type == "cuda":
                    logging.info(
                        f"GPU Memory after: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB"
                    )

                # Clear cache less frequently with large batches
                if device.type == "cuda" and batch_idx % 50 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logging.info(f"Error processing batch {batch_idx}: {e}")
                continue

    logging.info(f"Completed generation. Output written to {output_csv}")


if __name__ == "__main__":
    main()
