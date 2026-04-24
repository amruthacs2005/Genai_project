import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model (run once in terminal if not installed)
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple Inc. was founded by Steve Jobs in 1976. It is headquartered in Cupertino, California."

# -----------------------------
# 1. Tokenization
# -----------------------------
tokens = word_tokenize(text)
print("\nTokens:")
print(tokens)

# -----------------------------
# 2. Stop Word Removal
# -----------------------------
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]

print("\nAfter Stop Word Removal:")
print(filtered_tokens)

# -----------------------------
# 3. Stemming
# -----------------------------
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\nStemming:")
print(stemmed_words)

# -----------------------------
# 4. Lemmatization
# -----------------------------
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("\nLemmatization:")
print(lemmatized_words)

# -----------------------------
# 5. POS Tagging
# -----------------------------
pos_tags = nltk.pos_tag(tokens)

print("\nPOS Tagging:")
print(pos_tags)

# -----------------------------
# 6. Named Entity Recognition (NER)
# -----------------------------
doc = nlp(text)
print("\nNamed Entities:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

# -----------------------------
# 7. Word Frequency Analysis
# -----------------------------
word_freq = Counter(filtered_tokens)

print("\nWord Frequency:")
for word, freq in word_freq.items():
    print(word, ":", freq)



import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
