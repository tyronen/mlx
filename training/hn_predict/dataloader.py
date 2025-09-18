import torch
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location="cpu")

        self.features_num = data["features_num"].to(torch.float32)
        self.title_embeddings = data["title_embeddings"].to(torch.float32)
        self.domain_indices = data["domain_index"]
        self.tld_indices = data["tld_index"]
        self.user_indices = data["user_index"]

        self.targets = data["targets"]

        self.valid_indices = torch.arange(self.targets.shape[0])  # use full dataset

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return (
            self.features_num[real_idx],
            self.title_embeddings[real_idx],
            self.domain_indices[real_idx],
            self.tld_indices[real_idx],
            self.user_indices[real_idx],
            self.targets[real_idx],
        )
