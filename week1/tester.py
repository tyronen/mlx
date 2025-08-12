import torch
import sys

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <word> <model-file>")
    sys.exit(1)

filename = sys.argv[2] if len(sys.argv) > 2 else "embeddings/cbow_text8.pt"
checkpoint = torch.load(filename)
embeddings = checkpoint["embeddings"]
word_to_ix = checkpoint["word_to_ix"]
ix_to_word = checkpoint["ix_to_word"]


# Test semantic similarity
def find_similar(word, top_k=5):
    if word not in word_to_ix:
        return f"'{word}' not in vocabulary"

    idx = word_to_ix[word]
    word_vec = embeddings[idx]

    # Cosine similarity
    similarities = torch.cosine_similarity(word_vec.unsqueeze(0), embeddings)
    top_indices = similarities.topk(top_k + 1)[1][1:]  # Skip the word itself

    return [ix_to_word[i.item()] for i in top_indices]


def main(word):
    results = find_similar(word)
    suffix = ", ".join(results) if isinstance(results, list) else results
    print(f"{word}: {suffix}")


main(sys.argv[1])
