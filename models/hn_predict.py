import torch
import torch.nn as nn

DATASET_NAME = "OpenPipe/hacker-news"


class QuantileRegressionModel(nn.Module):
    def __init__(
        self,
        vector_size_num,
        scale,
        domain_vocab_size,
        tld_vocab_size,
        user_vocab_size,
        word_vocab_size,
        num_quantiles: int,
        word_embedding_dim=256,
        domain_embedding_dim=8,
        tld_embedding_dim=4,
        user_embedding_dim=8,
    ):
        super().__init__()

        self.num_quantiles = num_quantiles

        self.word_embedding = nn.Embedding(
            word_vocab_size, word_embedding_dim, padding_idx=0
        )
        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embedding_dim)
        self.tld_embedding = nn.Embedding(tld_vocab_size, tld_embedding_dim)
        self.user_embedding = nn.Embedding(user_vocab_size, user_embedding_dim)

        total_input_size = (
            vector_size_num
            + word_embedding_dim
            + domain_embedding_dim
            + tld_embedding_dim
            + user_embedding_dim
        )

        self.dropout = nn.Dropout(p=0.1)
        scaled_size = scale * total_input_size
        self.linear1 = nn.Linear(total_input_size, scaled_size)
        self.bn1 = nn.BatchNorm1d(scaled_size)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(scaled_size, scaled_size)
        self.bn2 = nn.BatchNorm1d(scaled_size)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(scaled_size, self.num_quantiles)

    def forward(self, features_num, title_idx, domain_idx, tld_idx, user_idx):
        # title_token_indices shape: [batch_size, sequence_length]
        title_emb = self.word_embedding(title_idx)
        mask = (title_idx != 0).unsqueeze(-1).float()  # [B, T, 1]
        masked = title_emb * mask
        sum_vec = masked.sum(dim=1)  # [B, E]
        len_vec = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
        title_vector = sum_vec / len_vec  # [B, E]
        # title_vector shape: [batch_size, lstm_hidden_dim * 2]
        # ---------------------------------------------
        domain_emb = self.domain_embedding(domain_idx)
        tld_emb = self.tld_embedding(tld_idx)
        user_emb = self.user_embedding(user_idx)

        full_input = torch.cat(
            [features_num, title_vector, domain_emb, tld_emb, user_emb], dim=1
        )

        x = self.dropout(full_input)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        out = self.linear3(x)
        return out  # shape: [batch_size, num_quantiles]


CACHE_FILE = "data/inference_cache.pkl"
