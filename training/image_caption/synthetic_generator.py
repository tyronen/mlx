from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch
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
        return self.image_paths[idx]


def generate_captions(device, model, processor, batch, profile=False):
    timings = {}

    # Load and process the images
    start = time.time()
    images = []
    for image_path in batch:
        images.append(Image.open(image_path).convert("RGB"))
    timings["image_loading"] = time.time() - start

    # Create per-image conversations (batched) for chat template
    start = time.time()
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
            }
        ]
        for image in images
    ]
    timings["message_creation"] = time.time() - start

    # Apply chat template (batched)
    start = time.time()
    texts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )
    timings["chat_template"] = time.time() - start

    # Process inputs
    start = time.time()
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    timings["input_processing"] = time.time() - start

    # Move to device
    start = time.time()
    inputs = inputs.to(device)
    timings["device_transfer"] = time.time() - start

    # Generate caption with detailed timing breakdown
    start = time.time()

    # Pre-generation setup timing
    setup_start = time.time()
    generation_kwargs = {
        "max_new_tokens": 128,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
        "num_beams": 1,
        "early_stopping": False,
    }
    # Prefer model-specific end token if available (e.g., <|im_end|> for chat)
    try:
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            isinstance(im_end_id, int)
            and im_end_id != processor.tokenizer.unk_token_id
            and im_end_id is not None
        ):
            generation_kwargs["eos_token_id"] = im_end_id
    except Exception:
        pass
    setup_time = time.time() - setup_start

    # Actual generation timing with CUDA sync for accuracy
    gen_start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    # Sync CUDA for accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    gen_time = time.time() - gen_start

    timings["generation"] = time.time() - start

    if profile:
        print(f"  Generation breakdown:")
        print(f"    Setup: {setup_time:.3f}s")
        print(f"    Model forward: {gen_time:.3f}s")
        print(f"    Total generation: {timings['generation']:.3f}s")
        print(
            f"    Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}"
        )
        print(f"    Generated shape: {generated_ids.shape}")
        print(
            f"    Generated length: {generated_ids.shape[1] - inputs['input_ids'].shape[1]} tokens"
        )

    # Decode the response
    start = time.time()
    # We only want to decode the newly generated tokens, not the prompt
    input_token_len = inputs["input_ids"].shape[1]
    generated_token_ids = generated_ids[:, input_token_len:]
    captions = processor.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
    )
    timings["decoding"] = time.time() - start

    if profile:
        total_time = sum(timings.values())
        print(f"\nBatch timing breakdown:")
        for stage, duration in timings.items():
            percentage = (duration / total_time) * 100
            print(f"  {stage}: {duration:.3f}s ({percentage:.1f}%)")
        print(f"  Total: {total_time:.3f}s")

    return captions


def main():
    utils.setup_logging()
    device = utils.get_device()

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load the model and processor with optimizations
    model = AutoModelForImageTextToText.from_pretrained(
        LLM,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation=(
            "flash_attention_2" if torch.cuda.is_available() else "eager"
        ),
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        use_safetensors=True,  # Use safetensors for faster loading
    )
    # This is the key: control vision tokens by setting pixel limits in the processor
    # This avoids creating >1400 patches per image, which was the bottleneck
    # These settings correspond to roughly 448x448 to 672x672 resolution.
    processor = AutoProcessor.from_pretrained(
        LLM,
        min_pixels=448 * 448,
        max_pixels=672 * 672,
    )

    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    coco_dataset = CocoDataset()

    # Use larger batch size - data shows larger batches are faster per image
    # Try to maximize GPU utilization while staying within memory limits
    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if gpu_memory > 20e9:  # > 20GB (RTX 5090)
            batch_size = 64  # Larger than 32, but not too large
        elif gpu_memory > 10e9:  # > 10GB
            batch_size = 32
        else:
            batch_size = 16
    else:
        batch_size = 16
    print(f"Using batch size: {batch_size}")
    num_workers = (
        6 if device.type == "cuda" else 0 if device.type == "mps" else 3
    )  # More workers
    dataloader = DataLoader(
        coco_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if device.type == "cuda" else None),  # Prefetch more batches
    )

    synthetic_dataset = dict()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating captions")):
        try:
            # Profile first 3 batches to identify bottlenecks
            profile_this_batch = batch_idx < 3
            if profile_this_batch:
                print(f"\n=== Profiling Batch {batch_idx} ===")
                if device.type == "cuda":
                    print(
                        f"GPU Memory before: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB"
                    )
            # We're still just testing accuracy
            if batch_idx > 1:
                break

            captions = generate_captions(
                device, model, processor, batch, profile=profile_this_batch
            )
            for image_path, caption in zip(batch, captions):
                synthetic_dataset[image_path] = caption

            if profile_this_batch and device.type == "cuda":
                print(
                    f"GPU Memory after: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB"
                )

            # Clear cache every 10 batches to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    output_csv = pathlib.Path("data/coco/metadata.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "text"])  # header row
        for img_path, caption in synthetic_dataset.items():
            writer.writerow([pathlib.Path(img_path).name, caption])

    print(f"Wrote {len(synthetic_dataset)} captions to {output_csv}")


if __name__ == "__main__":
    main()
