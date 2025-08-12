import torch
from torch import nn as nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, embeddings, embed_dim, dropout_rate):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.requires_grad = self.train_embeddings()
        self.gru = nn.GRU(
            embed_dim, embed_dim, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        emb = self.embedding(x)
        _, hidden = self.gru(emb)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        h = self.fc(hidden)
        dropout = self.dropout(h)
        return F.normalize(dropout, p=2, dim=1)


class QueryTower(Tower):
    def __init__(self, embeddings, embed_dim, dropout_rate):
        super().__init__(embeddings, embed_dim, dropout_rate)

    @classmethod
    def train_embeddings(cls):
        return True


class DocTower(Tower):
    def __init__(self, embeddings, embed_dim, dropout_rate):
        super().__init__(embeddings, embed_dim, dropout_rate)

    @classmethod
    def train_embeddings(cls):
        return True
