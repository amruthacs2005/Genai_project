import os
import sys
import requests
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 🔑 SET YOUR OPENAI API KEY HERE
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# IPC PDF details
IPC_URL = "https://www.mha.gov.in/sites/default/files/IPAct_1860.pdf"
IPC_FILE = "Indian_Penal_Code.pdf"

# ✅ Download PDF
def download_pdf(url, filename):
    print("Downloading IPC document...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print("Download error:", e)
        sys.exit(1)

# Download only if not present
if not os.path.exists(IPC_FILE):
    download_pdf(IPC_URL, IPC_FILE)

# ✅ Extract text from PDF
def extract_text(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print("Extraction error:", e)
        sys.exit(1)

print("Extracting text...")
ipc_text = extract_text(IPC_FILE)

if not ipc_text:
    print("Text extraction failed!")
    sys.exit(1)

# ✅ Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_text(ipc_text)

# ✅ Create embeddings and FAISS DB
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_texts(chunks, embeddings)

# ✅ Load chat model
llm = ChatOpenAI(model="gpt-4o-mini")

# ✅ Create QA system
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever()
)

# ✅ Chatbot loop
def chatbot():
    print("\n📘 IPC Chatbot (type 'exit' to quit)")
    while True:
        query = input("\nAsk: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        try:
            result = qa.invoke({"query": query})
            print("\nAnswer:", result["result"])
        except Exception as e:
            print("Error:", e)

# ✅ Run program
if __name__ == "__main__":
    chatbot()