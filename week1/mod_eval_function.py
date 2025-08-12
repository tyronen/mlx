import sys
import torch
import numpy as np

analogies = [
    ("python", "programming", "java", "programming"),  # Should be close
    ("microsoft", "windows", "apple", "macos"),
    ("javascript", "web", "python", "scripting"),
    ("google", "search", "facebook", "social"),
    ("linux", "open", "windows", "proprietary"),
    # General analogies (classic word2vec tests)
    ("king", "man", "queen", "woman"),
    ("paris", "france", "london", "england"),
    ("brother", "man", "sister", "woman"),
    ("big", "bigger", "small", "smaller"),
    ("good", "better", "bad", "worse"),

    # Science/Academic
    ("physics", "science", "history", "humanities"),
    ("doctor", "medicine", "lawyer", "law"),
    ("teacher", "school", "chef", "restaurant"),

    # Geography/Culture
    ("tokyo", "japan", "berlin", "germany"),
    ("dollar", "america", "euro", "europe"),
    ("english", "england", "french", "france"),
]

# word pairs that should be similar
similar_pairs = [
    ("computer", "machine"),
    ("programming", "coding"),
    ("internet", "web"),
    ("software", "program"),
    ("algorithm", "method"),
    ("javascript", "java"),
    ("microsoft", "apple"),
    # General semantic pairs
    ("car", "automobile"),
    ("happy", "joyful"),
    ("big", "large"),
    ("house", "home"),
    ("beautiful", "pretty"),
    ("smart", "intelligent"),
    ("fast", "quick"),

    # Conceptual pairs
    ("king", "queen"),
    ("man", "woman"),
    ("boy", "girl"),
    ("father", "mother"),
    ("brother", "sister"),

    # Academic/Professional
    ("doctor", "physician"),
    ("teacher", "professor"),
    ("student", "pupil"),
    ("book", "novel"),
    ("science", "research"),

    # Actions/Verbs
    ("walk", "run"),
    ("eat", "consume"),
    ("learn", "study"),
    ("write", "compose"),

    # Geography
    ("city", "town"),
    ("country", "nation"),
    ("mountain", "hill"),
    ("river", "stream"),
]

# Random word pairs that should be dissimilar
dissimilar_pairs = [
    ("computer", "tree"),
    ("programming", "music"),
    ("internet", "food"),
    ("algorithm", "color"),
    # Completely unrelated concepts
    ("happy", "database"),
    ("mountain", "programming"),
    ("music", "mathematics"),
    ("food", "politics"),
    ("color", "economics"),
    ("animal", "technology"),

    # Opposite concepts
    ("hot", "cold"),
    ("big", "small"),
    ("good", "bad"),
    ("fast", "slow"),
    ("happy", "sad"),
    ("light", "dark"),

    # Different domains
    ("science", "art"),
    ("medicine", "sports"),
    ("history", "technology"),
    ("music", "engineering"),
    ("literature", "mathematics"),
    ("politics", "biology"),

    # Abstract vs concrete
    ("love", "computer"),
    ("freedom", "table"),
    ("justice", "car"),
    ("beauty", "software"),
    ("truth", "phone"),
]

categories = {
    "programming": ["python", "java", "javascript", "programming", "coding"],
    "companies": ["google", "apple", "microsoft", "amazon", "facebook"],
    "tech_concepts": ["algorithm", "data", "software", "computer", "internet"],
    # General categories
    "colors": ["red", "blue", "green", "yellow", "black", "white"],
    "animals": ["cat", "dog", "bird", "fish", "horse", "cow"],
    "family": ["father", "mother", "brother", "sister", "son", "daughter"],
    "emotions": ["happy", "sad", "angry", "excited", "calm", "worried"],
    "body_parts": ["head", "hand", "foot", "eye", "nose", "mouth"],

    # Academic/Professional
    "sciences": ["physics", "chemistry", "biology", "mathematics", "psychology"],
    "professions": ["doctor", "teacher", "lawyer", "engineer", "artist"],
    "education": ["school", "university", "student", "professor", "learning"],

    # Geography/Places
    "countries": ["america", "england", "france", "germany", "japan"],
    "cities": ["london", "paris", "tokyo", "berlin", "moscow"],
    "geography": ["mountain", "river", "ocean", "forest", "desert"],

    # Arts/Culture
    "music": ["song", "piano", "guitar", "symphony", "opera"],
    "literature": ["book", "novel", "poem", "story", "author"],
    "sports": ["football", "basketball", "tennis", "swimming", "running"],

    # Time/Temporal
    "time_periods": ["day", "week", "month", "year", "century"],
    "seasons": ["spring", "summer", "autumn", "winter"],

    # Food/Consumption
    "food": ["bread", "meat", "fruit", "vegetable", "milk"],
    "cooking": ["cook", "bake", "fry", "boil", "grill"],

    # Transportation
    "vehicles": ["car", "train", "plane", "ship", "bicycle"],
    "transportation": ["travel", "journey", "road", "airport", "station"],

    # Abstract concepts
    "values": ["freedom", "justice", "truth", "beauty", "love"],
    "qualities": ["good", "bad", "beautiful", "ugly", "smart", "stupid"],
}


def word_analogy_test(embeddings, word_to_ix):
    """Test word analogies: king - man + woman = queen"""
    correct = 0
    total = 0

    total_sim = 0

    for a, b, c, expected in analogies:
        if all(word in word_to_ix for word in [a, b, c, expected]):
            # Get embeddings
            vec_a = embeddings[word_to_ix[a]]
            vec_b = embeddings[word_to_ix[b]]
            vec_c = embeddings[word_to_ix[c]]

            # Compute: c + (b - a)
            target_vec = vec_c + (vec_b - vec_a)

            # Find most similar word (excluding input words)
            best_similarity = -1
            best_word = None
            exclude = {a, b, c}

            for word, idx in word_to_ix.items():
                if word not in exclude:
                    similarity = torch.cosine_similarity(target_vec.unsqueeze(0),
                                                         embeddings[idx].unsqueeze(0))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_word = word

            if best_word == expected:
                correct += 1
            total += 1

            total_sim += best_similarity
            # print(f"{a} - {b} + {c} = {best_word} (expected: {expected})")

    accuracy = correct / total if total > 0 else 0
    avg_similarity_of_final_embedding = total_sim/len(analogies)
    # print(f"\nAnalogy Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy, avg_similarity_of_final_embedding

def semantic_similarity_test(embeddings, word_to_ix):
    """Test if semantically similar words have high cosine similarity"""

    similar_scores = []
    dissimilar_scores = []

    for w1, w2 in similar_pairs:
        if w1 in word_to_ix and w2 in word_to_ix:
            sim = torch.cosine_similarity(
                embeddings[word_to_ix[w1]].unsqueeze(0),
                embeddings[word_to_ix[w2]].unsqueeze(0)
            ).item()
            similar_scores.append(sim)
            # print(f"Similar: {w1} - {w2}: {sim:.3f}")

    for w1, w2 in dissimilar_pairs:
        if w1 in word_to_ix and w2 in word_to_ix:
            sim = torch.cosine_similarity(
                embeddings[word_to_ix[w1]].unsqueeze(0),
                embeddings[word_to_ix[w2]].unsqueeze(0)
            ).item()
            dissimilar_scores.append(sim)
            # print(f"Dissimilar: {w1} - {w2}: {sim:.3f}")

    avg_similar = np.mean(similar_scores) if similar_scores else 0
    avg_dissimilar = np.mean(dissimilar_scores) if dissimilar_scores else 0

    # print(f"\nAverage similar pairs similarity: {avg_similar:.3f}")
    # print(f"Average dissimilar pairs similarity: {avg_dissimilar:.3f}")
    # print(f"Separation score: {avg_similar - avg_dissimilar:.3f}")

    return avg_similar, avg_dissimilar

def category_clustering_test(embeddings, word_to_ix):
    """Test if words in same category cluster together"""

    category_scores = {}

    for category, words in categories.items():
        # Get embeddings for words in this category
        valid_words = [w for w in words if w in word_to_ix]
        if len(valid_words) < 2:
            continue

        category_embeddings = torch.stack([embeddings[word_to_ix[w]] for w in valid_words])

        # Calculate average pairwise similarity within category
        similarities = []
        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                sim = torch.cosine_similarity(
                    category_embeddings[i].unsqueeze(0),
                    category_embeddings[j].unsqueeze(0)
                ).item()
                similarities.append(sim)

        avg_similarity = np.mean(similarities)
        category_scores[category] = avg_similarity
        print(f"{category}: {avg_similarity:.3f}")

    overall_score = np.mean(list(category_scores.values()))
    # print(f"\nOverall clustering score: {overall_score:.3f}")
    return overall_score