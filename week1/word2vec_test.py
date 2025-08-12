import subprocess
import sys

test_categories = {
    "Programming Languages": ["python", "javascript", "rust", "go", "java"],
    "AI/ML": ["ai", "machine", "learning", "neural", "model"],
    "Startups": ["startup", "founder", "funding", "ipo", "unicorn"],
    "Big Tech": ["google", "apple", "microsoft", "amazon", "meta"],
    "Sentiment": ["amazing", "terrible", "broken", "innovative", "privacy"],
    "Tech Stack": ["react", "microservice", "node", "cloud", "api"],
    "Content Types": ["tutorial", "guide", "analysis", "review", "benchmark"],
    "Common Words": ["king", "man", "city", "water", "car"],
}

filename = sys.argv[1] if len(sys.argv) > 1 else "embeddings/cbow_text8.pt"

for category, words in test_categories.items():
    print(f"\n{'=' * 50}")
    print(f"Category: {category}")
    print("=" * 50)

    for word in words:
        try:
            subprocess.run([sys.executable, "tester.py", word, filename], check=True)
        except subprocess.CalledProcessError:
            print(f"'{word}' not found in vocabulary")
