# tokenize_hn_data.py
import re
import os
import sys
import html

titles_file = "data/titles.txt"
comments_file = "data/comments.txt"
output_file = "data/hn_corpus.txt"


def clean_text(text):
    """Clean and tokenize text similar to text8 format"""
    # Decode HTML entities first
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove apostrophe suffixes
    text = re.sub(r"n't\b", " not", text)  # "don't" -> "do not"
    text = re.sub(r"'[a-z]*\b", "", text)

    # Keep only letters, numbers, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\-\!]", " ", text)

    # Replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def tokenize_hn_data(output_file="hn_corpus.txt"):
    """Tokenize HN titles and comments into text8-like format"""

    print("Tokenizing HN data...")
    all_words = []

    # Process titles
    if os.path.exists(titles_file):
        print("Processing titles...")
        with open(titles_file, "r", encoding="utf-8") as f:
            title_count = 0
            for line in f:
                title = line.strip()
                if title:
                    cleaned = clean_text(title)
                    if cleaned:
                        words = cleaned.split()
                        all_words.extend(words)
                        title_count += 1

                        if title_count % 100000 == 0:
                            print(f"Processed {title_count} titles...")

        print(f"✅ Processed {title_count} titles")

    # Process comments
    if os.path.exists(comments_file):
        print("Processing comments...")
        with open(comments_file, "r", encoding="utf-8") as f:
            comment_count = 0
            for line in f:
                comment = line.strip()
                if comment:
                    cleaned = clean_text(comment)
                    if cleaned:
                        words = cleaned.split()
                        all_words.extend(words)
                        comment_count += 1

                        if comment_count % 10000 == 0:
                            print(f"Processed {comment_count} comments...")

        print(f"✅ Processed {comment_count} comments")

    # Write to output file (space-separated like text8)
    print(f"Writing {len(all_words)} words to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(all_words))

    # Calculate file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Created {output_file} ({file_size_mb:.1f} MB)")
    print(f"✅ Total words: {len(all_words):,}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_file = sys.argv[1]

    tokenize_hn_data(output_file)
