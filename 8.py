import os
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.document_loaders import GoogleDriveLoader


def load_text_from_google_drive(file_id):
    """
    Loads text from a Google Drive file and combines all pages into a single string.
    """
    loader = GoogleDriveLoader(file_id=file_id)
    documents = loader.load()
    if not documents:
        return ""
    # Combine all pages into one text
    return " ".join([doc.page_content for doc in documents])


def summarize_text(text):
    """
    Uses Cohere API to summarize the given text.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError(
            "Cohere API key not found. Please set the COHERE_API_KEY environment variable."
        )

    llm = Cohere(api_key=cohere_api_key)
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n{text}\n\nSummary:"
    )

    # Send the text to Cohere for summarization
    summary = llm(prompt_template.format(text=text))
    return summary


if __name__ == "__main__":
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your file ID
    try:
        text = load_text_from_google_drive(file_id)
        if text:
            summary = summarize_text(text)
            print("Summarized Text:")
            print(summary)
        else:
            print("Failed to load text from Google Drive.")
    except Exception as e:
        print(f"An error occurred: {e}")