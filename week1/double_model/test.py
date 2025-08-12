import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load("data/train.npz")
targets = data["targets"]

# # Replace zeros with 1 (your original step)
targets[targets==0] = 1

# # Count number of 1's and non-1's
num_ones = np.sum(targets == 1)
num_nonones = np.sum(targets != 1)

print(f"Number of 1 divided by non-1: {num_ones/num_nonones}")
print(f"Number of non-1: {num_nonones}")
print(f"Number of 1: {num_ones}")

# # Now filter out targets that are not equal to 1
# targets_nonone = targets[targets != 1]

# # Create bins for remaining values (from 2 to 10)
bins = np.arange(0, 12)  # 2 to 11 so that 10 is included

# # Use np.histogram to count occurrences
counts, _ = np.histogram(targets, bins=bins)

# Print counts for values 2-10
for i, count in zip(range(0, 11), counts):
    print(f"Targets == {i}: {count}")

# Plot
# plt.bar(range(2, 11), counts)
# plt.xlabel("Target value (excluding 1)")
# plt.ylabel("Count")
# plt.title("Target counts for values 2 to 10 (excluding 1)")
# plt.xticks(range(2, 11))
# plt.show()
