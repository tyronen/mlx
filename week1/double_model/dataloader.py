import torch
from torch.utils.data import Dataset
import numpy as np

class PrecomputedNPZDataset(Dataset):
    def __init__(self, npz_path, task="regression"):
        data = np.load(npz_path, allow_pickle=True)

        self.features_num = torch.tensor(data['features_num'], dtype=torch.float32)
        self.title_embeddings = torch.tensor(data['title_embeddings'], dtype=torch.float32)
        self.domain_indices = torch.tensor(data['domain_index'], dtype=torch.long)
        self.tld_indices = torch.tensor(data['tld_index'], dtype=torch.long)
        self.user_indices = torch.tensor(data['user_index'], dtype=torch.long)

        raw_targets = torch.tensor(data['targets'], dtype=torch.float32)
        shifted_targets = torch.clip(raw_targets, min=1) - 1

        if task == "classification":
            self.targets = (shifted_targets > 0).float()
            self.valid_indices = torch.arange(len(self.targets))  # use full dataset

        elif task == "regression":
            self.valid_indices = (shifted_targets > 0).nonzero(as_tuple=True)[0]  # use only non-zero samples
            self.targets = torch.log10(shifted_targets + 1)

        else:
            raise ValueError("task must be 'regression' or 'classification'")

    def __len__(self):
        return len(self.valid_indices)

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




