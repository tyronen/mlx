import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

DATASET_FILE = "data/datasets.pt"


def cosine_similarity(vec1, vec2):
    num = np.dot(vec1, vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return num / denom if denom != 0 else 0.0


class TripletDataset(Dataset):
    embed_cache = {}

    def __init__(self, ms_marco_data, tokenizer, device):
        self.triplets = []
        self.tokenizer = tokenizer
        self.device = device

        # Group by query to create triplets
        query_groups = {}
        for row in tqdm(ms_marco_data):
            query = row["query"]
            passages = row["passages"]
            passage_texts = passages["passage_text"]
            is_selected = np.array(passages["is_selected"])
            indices_ones = np.where(is_selected == 1)[0]
            indices_zeros = np.where(is_selected == 0)[0]
            if len(indices_ones) == 0 or len(indices_zeros) == 0:
                continue  # positive or negative is missing, skip this query
            best_similarity = 0
            best_pos = None
            best_neg = None
            for i in indices_ones:
                pos = passage_texts[i]
                vec_i = self.embed(pos)
                for j in indices_zeros:
                    neg = passage_texts[j]
                    vec_j = self.embed(neg)
                    similarity = cosine_similarity(vec_i, vec_j)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pos = pos
                        best_neg = neg
            query_groups[query] = {}
            query_groups[query]["pos"] = best_pos
            query_groups[query]["neg"] = best_neg

        # Create (query, positive_passage, negative_passage) triplets
        for query, pair in query_groups.items():
            if len(pair["pos"]) > 0 and len(pair["neg"]) > 0:
                self.triplets.append(
                    {
                        # token id tensors --------------------------------
                        "query": self.tokenize(query),
                        "positive": self.tokenize(pair["pos"]),
                        "negative": self.tokenize(pair["neg"]),
                        # raw texts ---------------------------------------
                        "query_text": query,
                        "positive_text": pair["pos"],
                        "negative_text": pair["neg"],
                    }
                )

    def embed(self, text):
        if text in self.embed_cache:
            return TripletDataset.embed_cache[text]
        embed = self.tokenizer.embed(text)
        TripletDataset.embed_cache[text] = embed
        return embed

    def tokenize(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        return self.tokenizer(text)["input_ids"].squeeze(0)

    def __getitem__(self, idx):
        return self.triplets[idx]

    def __len__(self):
        return len(self.triplets)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def replace_negatives(self, query_to_new_neg):
        """
        Replace negative passages with mined hard negatives.
        `query_to_new_neg`: dict  query_text -> new_negative_text
        """
        for trip in self.triplets:
            qtxt = trip["query_text"]
            if qtxt in query_to_new_neg:
                new_neg_txt = query_to_new_neg[qtxt]
                trip["negative_text"] = new_neg_txt
                trip["negative"] = self.tokenize(new_neg_txt)
