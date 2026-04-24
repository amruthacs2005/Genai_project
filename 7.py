from transformers import pipeline

def summarize_text(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


if __name__ == "__main__":
    passage = """
    Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data,
    adapt to new inputs, and perform human-like tasks.
    """

    summary = summarize_text(passage)
    print("Summarized Text:")
    print(summary)