"""
run_app.py
----------
Main entry point for running the RAG-based chatbot with Gradio web UI.
Now powered by Groq (free tier) — fastest & most reliable option in 2025.
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check for required dependencies
try:
    import gradio as gr
except ImportError:
    print("Error: 'gradio' module not found!")
    print("   Please install it with: pip install gradio")
    sys.exit(1)

import logging
from src.retriever.retreiver import Retriever  # fixed typo: retreiver → retriever
from src.retriever.vector_store import FAISSVectorStore
from src.llm.prompt_builder import PromptBuilder

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env automatically

groq_api_key = os.getenv("GROQ_API_KEY")

# Import Groq client (drop-in replacement)
from src.llm.llm_groq import GroqClient

# Try to import logger, fallback to standard logging
try:
    from src.utils.logger import logger
except (ImportError, AttributeError):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RAGChatbot")


class RAGChatbot:
    """RAG-based chatbot using Groq + FAISS retrieval."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm_client: GroqClient,
        top_k: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        self.retriever = Retriever(vector_store, top_k=top_k)
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        if not message or not message.strip():
            return "", history

        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(message)
            if retrieved_docs:
                context_text = "\n\n".join([
                    f"[Document {i+1}]: {doc.get('text', '')}"
                    for i, doc in enumerate(retrieved_docs)
                ])
                source_note = f"\n\n---\n*Based on {len(retrieved_docs)} retrieved document(s)*"
            else:
                context_text = "No relevant documents found in the knowledge base."
                source_note = "\n\n---\nWarning: No documents retrieved — answering from general knowledge"

            # Step 2: Build RAG prompt
            prompt = self.prompt_builder.build_qa_prompt(context=context_text, query=message)

            # Step 3: Generate response via Groq
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Step 4: Append to history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response + source_note})

            return "", history

        except Exception as e:
            error_msg = f"Error: Failed to generate response: {str(e)}"
            logger.error(f"Error in chat: {e}", exc_info=True)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history

    def clear_history(self):
        return []


def create_gradio_interface(chatbot: RAGChatbot) -> gr.Blocks:
    with gr.Blocks(title="RAG Chatbot (Powered by Groq)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# RAG Chatbot\nFast, accurate answers from your documents using **Groq** + **Qwen2.5/Llama-3.3**")

        chatbot_interface = gr.Chatbot(
            label="Conversation",
            height=600,
            show_copy_button=True,
            type="messages",
            avatar_images=("👤", "🤖")
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about your uploaded documents...",
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")

        # Event handlers
        msg_input.submit(chatbot.chat, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface])
        submit_btn.click(chatbot.chat, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot_interface, msg_input])

        gr.Markdown("""
        ### Tips
        - Be specific for best retrieval
        - Powered by **Groq** → near-instant responses
        - Uses state-of-the-art models: **Qwen2.5-72B**, **Llama-3.3-70B**, etc.
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot with Gradio UI — powered by Groq (free tier)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--index-path', type=str, default='data/faiss_index.bin',
                        help='Path to FAISS index (.bin)')
    parser.add_argument('--embed-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model (default works great)')
    parser.add_argument('--groq-api-key', type=str, default=groq_api_key,
                        help='Groq API key (or set GROQ_API_KEY env var)')
    parser.add_argument('--llm-model', type=str, default='llama-3.1-8b-instant',
                        help='Groq model (recommended: qwen2.5-72b-instruct, llama-3.3-70b-instruct)')
    parser.add_argument('--top-k', type=int, default=3, help='Number of documents to retrieve')
    parser.add_argument('--temperature', type=float, default=0.2, help='Response creativity')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Max output tokens')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    args = parser.parse_args()

    # Validate index exists
    if not os.path.exists(args.index_path):
        logger.error(f"FAISS index not found: {args.index_path}")
        logger.error("Run build_embeddings.py first!")
        sys.exit(1)

    # Load vector store
    vector_store = FAISSVectorStore(index_path=args.index_path, embedding_model=args.embed_model)
    vector_store.load()
    if len(vector_store.texts) == 0:
        logger.error("Vector store is empty! Rebuild your index.")
        sys.exit(1)

    # Validate Groq API key
    if not args.groq_api_key or args.groq_api_key.strip() == "":
        logger.error("ERROR: Groq API key is required!")
        logger.error("   → Get free key: https://console.groq.com/keys")
        logger.error("   → Then run with: --groq-api-key gsk_...")
        logger.error("   → Or set env: export GROQ_API_KEY=gsk_...")
        sys.exit(1)

    # Initialize Groq client
    logger.info(f"Initializing Groq client with model: {args.llm_model}")
    try:
        llm_client = GroqClient(
            model=args.llm_model,
            api_key=args.groq_api_key.strip(),
            max_retries=5
        )
        logger.info("SUCCESS: Groq client initialized — ready for lightning-fast RAG!")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        sys.exit(1)

    # Initialize RAG chatbot
    chatbot = RAGChatbot(
        vector_store=vector_store,
        llm_client=llm_client,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    # Launch Gradio UI
    demo = create_gradio_interface(chatbot)
    print(f"\nLaunching RAG Chatbot on http://{args.host}:{args.port}")
    if args.share:
        print("Public share link enabled!")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()