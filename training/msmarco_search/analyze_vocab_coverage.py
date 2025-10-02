#!/usr/bin/env python3
"""
Analyze vocabulary coverage between Word2Vec (text8) and MS MARCO datasets.
This helps understand how many MS MARCO terms are out-of-vocabulary.
"""

import logging
from collections import Counter
from typing import Any, Dict, List
from datasets import load_dataset
from models.msmarco_tokenizer import Word2VecTokenizer
from common import utils


def analyze_vocab_coverage():
    """Analyze vocabulary coverage between Word2Vec and MS MARCO"""
    utils.setup_logging()

    # Load Word2Vec tokenizer
    tokenizer = Word2VecTokenizer()
    word2vec_vocab = set(tokenizer.word_to_ix.keys())

    logging.info(f"Word2Vec vocabulary size: {len(word2vec_vocab)}")

    # Load MS MARCO data
    logging.info("Loading MS MARCO dataset...")
    ms_marco_data = load_dataset("ms_marco", "v1.1")

    # Collect all unique words from MS MARCO
    ms_marco_words = set()
    total_tokens = 0
    oov_tokens = 0

    for split in ["train", "validation", "test"]:
        logging.info(f"Processing {split} split...")
        for row in ms_marco_data[split]:
            # Process query
            row_dict = dict(row)  # Convert to dict for proper type handling
            query_words = str(row_dict["query"]).lower().split()
            for word in query_words:
                ms_marco_words.add(word)
                total_tokens += 1
                if word not in word2vec_vocab:
                    oov_tokens += 1

            # Process passages
            passages = row_dict["passages"]
            if isinstance(passages, dict) and "passage_text" in passages:
                passage_texts = passages["passage_text"]
                if isinstance(passage_texts, list):
                    for passage_text in passage_texts:
                        passage_words = str(passage_text).lower().split()
                        for word in passage_words:
                            ms_marco_words.add(word)
                            total_tokens += 1
                            if word not in word2vec_vocab:
                                oov_tokens += 1

    # Calculate coverage statistics
    ms_marco_vocab_size = len(ms_marco_words)
    covered_words = ms_marco_words.intersection(word2vec_vocab)
    uncovered_words = ms_marco_words - word2vec_vocab

    coverage_ratio = len(covered_words) / ms_marco_vocab_size
    token_coverage_ratio = 1.0 - (oov_tokens / total_tokens)

    logging.info("=" * 60)
    logging.info("VOCABULARY COVERAGE ANALYSIS")
    logging.info("=" * 60)
    logging.info(f"Word2Vec vocabulary size: {len(word2vec_vocab):,}")
    logging.info(f"MS MARCO vocabulary size: {ms_marco_vocab_size:,}")
    logging.info(f"Covered words: {len(covered_words):,} ({coverage_ratio:.2%})")
    logging.info(f"Uncovered words: {len(uncovered_words):,} ({1-coverage_ratio:.2%})")
    logging.info(f"Token coverage: {token_coverage_ratio:.2%}")
    logging.info(f"OOV tokens: {oov_tokens:,} out of {total_tokens:,}")

    # Show most frequent uncovered words
    logging.info("\nMost frequent uncovered words:")
    uncovered_counter = Counter()
    for split in ["train", "validation", "test"]:
        for row in ms_marco_data[split]:
            row_dict = dict(row)  # Convert to dict for proper type handling
            for word in str(row_dict["query"]).lower().split():
                if word not in word2vec_vocab:
                    uncovered_counter[word] += 1

            passages = row_dict["passages"]
            if isinstance(passages, dict) and "passage_text" in passages:
                passage_texts = passages["passage_text"]
                if isinstance(passage_texts, list):
                    for passage_text in passage_texts:
                        for word in str(passage_text).lower().split():
                            if word not in word2vec_vocab:
                                uncovered_counter[word] += 1

    for word, count in uncovered_counter.most_common(20):
        logging.info(f"  {word}: {count:,}")

    # Recommendations
    logging.info("\n" + "=" * 60)
    logging.info("RECOMMENDATIONS")
    logging.info("=" * 60)

    if coverage_ratio < 0.8:
        logging.info(
            "❌ LOW VOCABULARY COVERAGE - Consider training Word2Vec on MS MARCO data"
        )
    elif coverage_ratio < 0.9:
        logging.info(
            "⚠️  MODERATE VOCABULARY COVERAGE - May benefit from MS MARCO-trained embeddings"
        )
    else:
        logging.info("✅ GOOD VOCABULARY COVERAGE - Current approach should work well")

    if token_coverage_ratio < 0.9:
        logging.info("❌ HIGH OOV RATE - Many tokens will be mapped to <UNK>")
    else:
        logging.info("✅ LOW OOV RATE - Most tokens have embeddings")

    return {
        "word2vec_vocab_size": len(word2vec_vocab),
        "ms_marco_vocab_size": ms_marco_vocab_size,
        "coverage_ratio": coverage_ratio,
        "token_coverage_ratio": token_coverage_ratio,
        "oov_tokens": oov_tokens,
        "total_tokens": total_tokens,
        "uncovered_words": list(uncovered_words)[:100],  # First 100 for inspection
    }


if __name__ == "__main__":
    analyze_vocab_coverage()
