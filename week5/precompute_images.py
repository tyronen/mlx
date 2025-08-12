import os
from tqdm import tqdm
import models
import torch
from torch.utils.data import Dataset, DataLoader
import utils

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
    model = models.VitEncoder()
    model.eval()

    imagepath, image_filenames, _ = utils.get_captions()

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
    output_dir = os.path.dirname(models.IMAGES_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(models.IMAGES_PATH):
        os.remove(models.IMAGES_PATH)
    torch.save(features, models.IMAGES_PATH)
    print(f"Saved {len(features)} image embeddings to {models.IMAGES_PATH}")


if __name__ == "__main__":
    main()
