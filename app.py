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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
st.title("🤖 AI Document Q&A Assistant")

st.markdown(
    "**Upload a PDF or text file → Ask unlimited questions → Get AI-powered answers**"
)

st.markdown("---")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:

    st.header("ℹ️ How It Works")

    st.markdown("""
1. Upload your document
2. AI creates embeddings
3. FAISS stores document vectors
4. Ask unlimited questions
5. RAG retrieves relevant information
6. Groq LLM generates answers
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


    st.markdown("---")


    if st.button("🗑 Clear Chat"):

        st.session_state.chat_history = []

        st.success("Chat history cleared!")


# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your document",
    type=["pdf", "txt"]
)


if uploaded_file is not None:


    # Process only new files
    if st.session_state.file_name != uploaded_file.name:


        suffix = (
            ".pdf"
            if uploaded_file.name.endswith(".pdf")
            else ".txt"
        )


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(uploaded_file.read())

            temp_path = tmp.name



        with st.spinner("🔄 Processing document..."):


            try:

                st.session_state.vector_store = (
                    load_and_index_document(temp_path)
                )


                st.session_state.file_name = uploaded_file.name


                # Clear old conversation for new document
                st.session_state.chat_history = []


                st.success(
                    f"✅ {uploaded_file.name} indexed successfully!"
                )


            except Exception as e:

                st.error(
                    f"Error processing document: {e}"
                )

                st.stop()


            finally:

                os.unlink(temp_path)



    else:

        st.success(
            f"✅ {uploaded_file.name} already indexed."
        )



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

            st.warning(
                "Please enter a question."
            )


        else:


            with st.spinner("🤖 Thinking..."):


                try:

                    answer, sources = get_answer(
                        st.session_state.vector_store,
                        question
                    )


                    # Store conversation

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": sources
                        }
                    )


                except Exception as e:

                    st.error(
                        f"Error generating answer: {e}"
                    )



    # --------------------------------------------------------
    # Chat History
    # --------------------------------------------------------

    if st.session_state.chat_history:


        st.markdown("---")

        st.subheader("💬 Conversation History")


        for chat in reversed(
            st.session_state.chat_history
        ):


            st.markdown(
                f"👤 **You:** {chat['question']}"
            )


            st.success(
                f"🤖 **AI:** {chat['answer']}"
            )


            if chat["sources"]:


                with st.expander(
                    "📚 Source Documents"
                ):

                    for source in chat["sources"]:

                        st.write(
                            f"📄 {source}"
                        )


            st.markdown("---")



    # --------------------------------------------------------
    # Example Questions
    # --------------------------------------------------------

    st.subheader("💡 Example Questions")


    st.markdown("""
- What is the main topic?
- Summarize the document.
- What are the key points?
- Explain the conclusion.
- What does the document say about ______?
""")


else:


    st.info(
        "👆 Upload a PDF or TXT file to begin."
    )


    st.markdown("---")


    st.subheader(
        "🎯 Supported Documents"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("✅ Research Papers")
        st.markdown("✅ Technical Manuals")
        st.markdown("✅ Company Reports")


    with col2:

        st.markdown("✅ Legal Documents")
        st.markdown("✅ Employee Handbooks")
        st.markdown("✅ Study Notes")