# RAG-based Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that combines a vector-based retriever with an LLM to deliver accurate, grounded responses using your domain data.

## Features
- Retrieval-Augmented responses grounded in your documents
- Modular pipeline: ingest → chunk → embed → retrieve → generate
- Pluggable vector stores (e.g., Chroma/FAISS) and embedding models (SentenceTransformers)
- Local LLM via Ollama or remote providers
- Gradio web UI and API backend

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-github-username/your-repo-name.git
   cd your-repo-name
   ```
2. Create and activate a virtual environment:
   - Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Ingest data and build embeddings:
  ```bash
  python scripts/build_embeddings.py
  ```
- Run the chatbot application:
  - Gradio web UI:
    ```bash
    python scripts/run_app.py
    ```
  - API (FastAPI+Uvicorn):
    ```bash
    uvicorn src.interface.app_api:app --reload
    ```

Configure `.env` and `config.yaml` as needed for your environment and data.

---

For more details on architecture and advanced configuration, see comments in code and module docstrings.
