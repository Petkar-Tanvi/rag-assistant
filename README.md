# 🤖 AI Document Q&A Assistant (RAG-based)

Upload any PDF or text document and ask questions in plain English.
AI retrieves the most relevant sections and generates precise answers.

## Tech Stack
- **LangChain** — RAG orchestration
- **FAISS** — Vector similarity search
- **Groq / Llama 3** — Free LLM
- **HuggingFace** — Text embeddings
- **Streamlit** — Web interface

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
## 🚀 Live Demo

Try the application here:

https://rag-assistant-tanvi.streamlit.app/