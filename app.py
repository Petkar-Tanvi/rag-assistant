# ============================================================
# AI Document Q&A Assistant — Streamlit UI
# Tech: RAG + LangChain + Groq + FAISS
# ============================================================

import streamlit as st
import tempfile
import os
from rag_engine import load_and_index_document, get_answer

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered"
)

# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.title("🤖 AI Document Q&A Assistant")
st.markdown("**Upload a PDF or text file → Ask unlimited questions → Get AI-powered answers**")
st.markdown("---")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ How It Works")

    st.markdown("""
1. Upload your document
2. AI processes and indexes it
3. Ask multiple questions
4. RAG retrieves relevant chunks
5. Groq LLM generates accurate answers
""")

    st.markdown("---")

    st.subheader("🛠 Tech Stack")
    st.markdown("""
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq Llama
- Streamlit
""")

# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your document",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    # Process only if it's a new file
    if st.session_state.file_name != uploaded_file.name:

        suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        with st.spinner("🔄 Processing document..."):

            try:
                st.session_state.vector_store = load_and_index_document(temp_path)
                st.session_state.file_name = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} indexed successfully!")

            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.stop()

            finally:
                os.unlink(temp_path)

    else:
        st.success(f"✅ {uploaded_file.name} already indexed.")

    st.markdown("---")

    # --------------------------------------------------------
    # Question Section
    # --------------------------------------------------------
    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Type your question:",
        placeholder="Example: What is the main topic of this document?"
    )

    if st.button("🔍 Get Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:
            with st.spinner("🤖 Thinking..."):

                try:
                    answer = get_answer(
                        st.session_state.vector_store,
                        question
                    )

                    st.markdown("### 📝 Answer")
                    st.success(answer)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    st.markdown("---")

    st.subheader("💡 Example Questions")

    st.markdown("""
- What is the main topic?
- Summarize the document.
- What are the key points?
- Explain the conclusion.
- What does the document say about ______?
""")

else:

    st.info("👆 Upload a PDF or TXT file to begin.")

    st.markdown("---")

    st.subheader("🎯 Supported Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("✅ Research Papers")
        st.markdown("✅ Technical Manuals")
        st.markdown("✅ Company Reports")

    with col2:
        st.markdown("✅ Legal Documents")
        st.markdown("✅ Employee Handbooks")
        st.markdown("✅ Study Notes")