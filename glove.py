import numpy as np
import pandas as pd
from collections import defaultdict

# Sample corpus
text = "the king loves the queen and the queen loves the king"
tokens = text.lower().split()

window_size = 2

# Build vocabulary
vocab = sorted(set(tokens))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

# Initialize co-occurrence matrix
co_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

# Build co-occurrence counts
for i, word in enumerate(tokens):
    word_id = word_to_id[word]

    start = max(i - window_size, 0)
    end = min(i + window_size + 1, len(tokens))

    for j in range(start, end):
        if i != j:
            context_word = tokens[j]
            context_id = word_to_id[context_word]
            co_matrix[word_id][context_id] += 1

# Convert to table
df = pd.DataFrame(co_matrix, index=vocab, columns=vocab)

print("Co-occurrence Matrix (Window size = 2):")
print(df)
