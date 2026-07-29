# ============================================================
# RAG Engine — Core Logic
# Retrieval Augmented Generation using LangChain + Groq + FAISS
# ============================================================

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

print("API Key:", os.getenv("GROQ_API_KEY"))

def load_and_index_document(file_path: str):
    """
    Step 1: Load document
    Step 2: Split into chunks
    Step 3: Convert to vectors (embeddings)
    Step 4: Store in FAISS vector database
    """

    # Load document based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split into small chunks — AI works better with smaller pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # each chunk = 500 characters
        chunk_overlap=50      # 50 char overlap so meaning isn't lost
    )
    chunks = splitter.split_documents(documents)

    # Convert text chunks to vectors using a free embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # lightweight, fast, free
    )

    # Store vectors in FAISS (Facebook AI Similarity Search)
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


def get_answer(vector_store, question: str) -> str:
    """
    Step 1: Search vector store for relevant chunks
    Step 2: Pass chunks + question to LLM
    Step 3: LLM generates answer based on document context
    """

    # Initialize Groq LLM (Llama 3 — free and fast)
    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
    

    # Create RAG chain: retriever finds docs, LLM answers
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",       # 'stuff' = put all chunks into prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

    response = qa_chain.invoke({"query": question})
    return response["result"]