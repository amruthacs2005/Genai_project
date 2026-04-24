# Required Libraries
import gensim.downloader as api
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained Word2Vec model
print("Loading pre-trained Word2Vec model...")
model = api.load('word2vec-google-news-300')

# List of sports-related words
sports_words = [
    'soccer', 'football', 'basketball', 'player', 'team',
    'coach', 'referee', 'goal', 'championship', 'league'
]

# Get word vectors
word_vectors = [model[word] for word in sports_words]

# Dimensionality reduction (PCA)
print("Performing dimensionality reduction...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# ✅ FIXED INDENTATION HERE
for i, word in enumerate(sports_words):
    plt.text(pca_result[i, 0] + 0.01,
             pca_result[i, 1] + 0.01,
             word,
             fontsize=12)

plt.title("2D Visualization of Sports Word Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Function to find similar words
def find_similar_words(input_word, top_n=5):
    try:
        similar_words = model.most_similar(input_word, topn=top_n)
        print(f"\nMost similar words to '{input_word}':")
        for word, similarity in similar_words:
            print(f"{word}: {similarity}")
    except KeyError:
        print(f"Word '{input_word}' not found.")

# Test
find_similar_words('soccer')