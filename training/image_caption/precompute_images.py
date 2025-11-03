import os
import argparse
from tqdm import tqdm
from models import image_caption, image_caption_utils
import torch
from torch.utils.data import Dataset, DataLoader
from common import arguments
from PIL import Image


class ImageFeatureDataset(Dataset):
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


def main(dataset_name, test_mode):
    model = image_caption.ImageEncoder()
    model.eval()

    if dataset_name == "flickr":
        imagepath, image_filenames, _ = image_caption_utils.get_flickr(test_mode)
        output_path = image_caption.FLICKR_FEATURES_PATH
    elif dataset_name == "coco":
        imagepath, image_filenames, _ = image_caption_utils.get_coco(test_mode)
        output_path = image_caption.COCO_FEATURES_PATH
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'coco' or 'flickr'.")

    # Dataset and DataLoader
    dataset = ImageFeatureDataset(image_filenames, imagepath)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    features = {}

    for filenames, images in tqdm(dataloader):
        outputs = model(images).cpu()
        for fname, vec in zip(filenames, outputs):
            features[fname] = vec

    # Save
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    torch.save(features, output_path)
    print(f"Saved {len(features)} image embeddings to {output_path}")


if __name__ == "__main__":
    parser = arguments.get_parser(
        description="Precompute image embeddings for image captioning"
    )
    args = parser.parse_args()
    main(args.dataset, args.check)
