import os
from tqdm import tqdm
from models import image_caption, image_caption_utils
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image


class ViTFeatureDataset(Dataset):
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


def main():
    # Load ViT
    model = image_caption.ImageEncoder()
    model.eval()

    imagepath, image_filenames, _ = image_caption_utils.get_captions()

    # Dataset and DataLoader
    dataset = ViTFeatureDataset(image_filenames, f"{imagepath}/Images")
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    features = {}

    for filenames, images in tqdm(dataloader):
        outputs = model(images).cpu().numpy()
        for fname, vec in zip(filenames, outputs):
            features[fname] = vec

    # Save
    # Ensure the output directory exists
    output_dir = os.path.dirname(image_caption.IMAGES_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(image_caption.IMAGES_PATH):
        os.remove(image_caption.IMAGES_PATH)
    torch.save(features, image_caption.IMAGES_PATH)
    print(f"Saved {len(features)} image embeddings to {image_caption.IMAGES_PATH}")


if __name__ == "__main__":
    main()
