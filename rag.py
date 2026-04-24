from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

# 🔑 Set API Key
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# ✅ Step 1: Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# ✅ Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# ✅ Step 3: Create embeddings
embeddings = OpenAIEmbeddings()

# ✅ Step 4: Store in FAISS
vector_store = FAISS.from_documents(docs, embeddings)

# ✅ Step 5: Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ✅ Step 6: Create RAG pipeline
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# ✅ Step 7: Ask query
query = "What is the main topic of the document?"
result = qa_chain.invoke({"query": query})

print("Answer:", result["result"])