import math

import torch

WORD2VEC_FILE = "data/word2vec_skipgram.pth"
MAX_LENGTH = 128


class Word2VecTokenizer:
    def __init__(self, doc_freq=None):
        checkpoint = torch.load(WORD2VEC_FILE, weights_only=False)
        self.embeddings = checkpoint["embeddings"]
        self.word_to_ix = checkpoint["word_to_ix"]
        self.ix_to_word = checkpoint["ix_to_word"]
        self.doc_freq = doc_freq

        # Add special tokens if not present
        if "<PAD>" not in self.word_to_ix:
            # Shift all existing indices up by 2 to make room for special tokens
            old_word_to_ix = self.word_to_ix.copy()

            self.word_to_ix = {"<PAD>": 0, "<UNK>": 1}
            self.ix_to_word = {0: "<PAD>", 1: "<UNK>"}

            # Reindex existing vocabulary
            for word, old_idx in old_word_to_ix.items():
                new_idx = old_idx + 2
                self.word_to_ix[word] = new_idx
                self.ix_to_word[new_idx] = word
            embedding_dim = self.embeddings.shape[1]  # Assuming [vocab_size, dim]
            pad_vec = torch.zeros((1, embedding_dim))
            unk_vec = torch.randn((1, embedding_dim))
            self.embeddings = torch.cat([pad_vec, unk_vec, self.embeddings], dim=0)

        self.vocab_size = len(self.word_to_ix)
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"

    def embed(self, text):
        words = text.lower().split()[:MAX_LENGTH]
        vec_sum, weight_sum = 0.0, 0.0
        for w in words:
            idx = self.word_to_ix.get(w, self.word_to_ix["<UNK>"])
            idf = 1.0 / math.log1p(self.doc_freq.get(w, 1))
            vec_sum += self.embeddings[idx] * idf
            weight_sum += idf
        return (
            (vec_sum / weight_sum).cpu().numpy()
            if weight_sum
            else self.embeddings[self.word_to_ix["<UNK>"]]
        )

    # tokenize() calls this
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        tokenized_batch = []
        for text in texts:
            # Simple whitespace tokenization
            words = text.lower().split()[:MAX_LENGTH]

            # Convert to indices
            indices = [
                self.word_to_ix.get(word, self.word_to_ix.get(self.unk_token, 1))
                for word in words
            ]

            # Pad to max_length
            indices = indices[:MAX_LENGTH]
            indices += [self.word_to_ix[self.pad_token]] * (MAX_LENGTH - len(indices))

            tokenized_batch.append(indices)

        return {"input_ids": torch.tensor(tokenized_batch, dtype=torch.long)}
