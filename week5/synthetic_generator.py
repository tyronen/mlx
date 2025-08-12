from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import pathlib
import csv
from torch.utils.data import Dataset
import utils
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


def generate_captions(device, model, processor, batch):
    # Load and process the images
    images = []
    for image_path in batch:
        images.append(Image.open(image_path))

    # Create a conversation format (Qwen expects chat format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
        for image in images
    ]

    # Apply chat template
    texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Decode the response
    captions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return captions


def main():
    utils.setup_logging()
    device = utils.get_device()
    # Load the model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        LLM, torch_dtype=torch.float16, device_map="auto"
    ).to(device)
    processor = AutoProcessor.from_pretrained(LLM)

    coco_dataset = CocoDataset()
    dataloader = utils.CustomDataLoader(coco_dataset, device, batch_size=64)

    synthetic_dataset = dict()
    for batch in tqdm(dataloader):
        captions = generate_captions(device, model, processor, batch)
        for image_path, caption in zip(batch, captions):
            synthetic_dataset[image_path] = caption

    output_csv = pathlib.Path("data/coco/metadata.csv")
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "text"])  # header row
        for img_path, caption in synthetic_dataset.items():
            writer.writerow([pathlib.Path(img_path).name, caption])

    print(f"Wrote {len(synthetic_dataset)} captions to {output_csv}")


if __name__ == "__main__":
    main()
