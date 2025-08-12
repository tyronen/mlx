import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FullModel(nn.Module):
    def __init__(self, 
                 vector_size_num, 
                 vector_size_title, 
                 scale,
                 domain_vocab_size, 
                 tld_vocab_size, 
                 user_vocab_size,
                 domain_embedding_dim=8, 
                 tld_embedding_dim=4,
                 user_embedding_dim=8):
        super().__init__()

        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embedding_dim)
        self.tld_embedding = nn.Embedding(tld_vocab_size, tld_embedding_dim)
        self.user_embedding = nn.Embedding(user_vocab_size, user_embedding_dim)

        total_input_size = vector_size_num + vector_size_title + domain_embedding_dim + tld_embedding_dim + user_embedding_dim

        self.linear1 = nn.Linear(total_input_size, scale * total_input_size)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(scale * total_input_size, scale * total_input_size)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(scale * total_input_size, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features_num, title_emb, domain_idx, tld_idx, user_idx):
        domain_emb = self.domain_embedding(domain_idx)
        tld_emb = self.tld_embedding(tld_idx)
        user_emb = self.user_embedding(user_idx)

        full_input = torch.cat([features_num, title_emb, domain_emb, tld_emb, user_emb], dim=-1)

        x = self.linear1(full_input)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.relu2(x)

        out = self.linear3(x)
        
        return out

