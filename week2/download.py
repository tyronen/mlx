from datasets import load_dataset

# Load MS MARCO passage ranking dataset
dataset = load_dataset("ms_marco", "v1.1")

# See what fields are available
print(dataset['train'].features)
print(dataset['train'][0].keys())

# Typical structure:
# - query: "what is the population of seattle"
# - passages: list of candidate passages
# - positive_passages: relevant passages
# - negative_passages: irrelevant passages