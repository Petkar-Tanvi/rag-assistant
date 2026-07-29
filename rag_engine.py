# ============================================================
# RAG Engine — Core Logic
# Retrieval Augmented Generation using LangChain + Groq + FAISS
# ============================================================

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_classic.chains import RetrievalQA

from dotenv import load_dotenv

import os


# Load environment variables
load_dotenv()


# ------------------------------------------------------------
# Load Document + Create Vector Store
# ------------------------------------------------------------

def load_and_index_document(file_path: str):

    """
    1. Load document
    2. Split into chunks
    3. Create embeddings
    4. Store vectors in FAISS
    """


    # Load document

    if file_path.endswith(".pdf"):

        loader = PyPDFLoader(file_path)

    else:

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )


    documents = loader.load()



    # Split documents into chunks

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50

    )


    chunks = splitter.split_documents(documents)



    # Create embeddings

    embeddings = HuggingFaceEmbeddings(

        model_name="all-MiniLM-L6-v2"

    )



    # Create FAISS vector database

    vector_store = FAISS.from_documents(

        chunks,

        embeddings

    )


    return vector_store



# ------------------------------------------------------------
# Generate Answer + Sources
# ------------------------------------------------------------

def get_answer(vector_store, question: str):

    """
    RAG pipeline:

    Question
        |
        ↓
    FAISS Retriever
        |
        ↓
    Relevant document chunks
        |
        ↓
    Groq LLM
        |
        ↓
    Answer + citations
    """



    # Initialize Groq LLM

    llm = ChatGroq(

        model="llama-3.1-8b-instant",

        temperature=0.2,

        groq_api_key=os.getenv(
            "GROQ_API_KEY"
        )

    )



    # Create RAG chain

    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        chain_type="stuff",

        retriever=vector_store.as_retriever(

            search_kwargs={
                "k": 3
            }

        ),

        return_source_documents=True

    )



    # Get response

    response = qa_chain.invoke(

        {
            "query": question
        }

    )



    answer = response["result"]



    # Extract sources

    sources = []


    for doc in response["source_documents"]:


        source_file = doc.metadata.get(

            "source",

            "Unknown document"

        )


        page_number = doc.metadata.get(

            "page",

            None

        )


        if page_number is not None:

            sources.append(

                f"{source_file} - Page {page_number + 1}"

            )

        else:

            sources.append(

                source_file

            )



    # Remove duplicate sources

    sources = list(

        set(sources)

    )


    return answer, sources