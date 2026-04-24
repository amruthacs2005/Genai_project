from transformers import pipeline

# Load the sentiment analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of the given text using a pre-trained model."""
    result = sentiment_pipeline(text)[0]   # Get the first result
    return result   # Returns dictionary with label and score

# Example sentences for real-world application
feedbacks = [
    "The product is amazing! I love it.",
    "It was an average experience, nothing special.",
    "I'm extremely disappointed with the service.",
    "The food was okay, but the service was slow.",
    "I had a fantastic time at the hotel. Everything was perfect!"
]

for feedback in feedbacks:
    sentiment = analyze_sentiment(feedback)
    print(f"Text: {feedback}")
    print(f"Sentiment: {sentiment['label']}, Confidence: {sentiment['score']:.2f}\n")