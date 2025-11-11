"""
run_app.py
----------
Main entry point for running the RAG-based chatbot with a Gradio web UI.

This script initializes the RAG pipeline components (retriever, LLM client, prompt builder)
and launches an interactive Gradio interface for chatting with the document-based assistant.
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional, Dict

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import logging
import requests
from src.retriever.retreiver import Retriever
from src.retriever.vector_store import FAISSVectorStore
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_builder import PromptBuilder

# Try to import logger, fallback to standard logging if it fails
try:
    from src.utils.logger import logger
except (ImportError, AttributeError):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RAGChatbot")


class RAGChatbot:
    """RAG-based chatbot that combines retrieval with LLM generation."""
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm_model: str = "llama2",
        top_k: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            vector_store: Initialized vector store instance
            llm_model: Ollama model name (e.g., "llama2", "gemma:2b")
            top_k: Number of retrieved documents to use as context
            temperature: LLM temperature for generation
            max_tokens: Maximum tokens in LLM response
        """
        self.retriever = Retriever(vector_store, top_k=top_k)
        self.llm_client = OllamaClient(model=llm_model)
        self.prompt_builder = PromptBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_history: List[Tuple[str, str]] = []
        
    def chat(self, message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process a user message and generate a response using RAG.
        
        Args:
            message: User's input message
            history: Chat history in Gradio messages format [{"role": "user", "content": "..."}, ...]
            
        Returns:
            Updated message (empty string) and updated history
        """
        if not message or not message.strip():
            return "", history
        
        try:
            # Retrieve relevant context
            logger.info(f"Retrieving context for query: {message[:50]}...")
            retrieved_docs = self.retriever.retrieve(message)
            
            # Debug: Log retrieval results with full details
            logger.info(f"✅ Retrieved {len(retrieved_docs)} documents for query: '{message}'")
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs):
                    text_preview = doc.get('text', '')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', '')
                    score = doc.get('score', 'N/A')
                    metadata = doc.get('metadata', {})
                    logger.info(f"  📄 Document {i+1}: Score={score:.4f}, Source={metadata.get('source', 'unknown')}")
                    logger.info(f"     Preview: {text_preview}")
            else:
                logger.warning("⚠️  NO DOCUMENTS RETRIEVED - Vector store might be empty or query doesn't match")
                logger.warning(f"   Vector store has {len(self.retriever.vector_store.texts)} documents loaded")
            
            # Combine retrieved contexts
            if retrieved_docs:
                context_text = "\n\n".join([
                    f"[Context {i+1}]: {doc.get('text', '')}" 
                    for i, doc in enumerate(retrieved_docs)
                ])
                logger.info(f"✅ Using {len(retrieved_docs)} relevant documents as context (total context length: {len(context_text)} chars)")
            else:
                context_text = "No relevant context found in the knowledge base."
                logger.warning("⚠️  No documents retrieved for query - response will be based on LLM knowledge only")
            
            # Build prompt with context
            prompt = self.prompt_builder.build_qa_prompt(
                context=context_text,
                query=message
            )
            
            # Log prompt preview to verify context is included
            logger.info(f"📝 Prompt length: {len(prompt)} characters")
            if retrieved_docs:
                logger.info(f"   Context included: {len(context_text)} characters from documents")
                logger.debug(f"   Prompt preview (first 300 chars): {prompt[:300]}...")
            else:
                logger.warning("   ⚠️  NO CONTEXT IN PROMPT - Only LLM knowledge will be used")
            
            # Generate response using LLM
            logger.info("🤖 Generating response with LLM...")
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Update history with new messages format
            history.append({"role": "user", "content": message})
            
            # Add retrieval info to response if documents were found
            if retrieved_docs:
                response_with_info = f"{response}\n\n---\n*Based on {len(retrieved_docs)} document(s) from your knowledge base*"
            else:
                response_with_info = f"{response}\n\n---\n*⚠️ No documents retrieved - response based on LLM general knowledge only*"
            
            history.append({"role": "assistant", "content": response_with_info})
            
            logger.info(f"✅ Response generated successfully (length: {len(response)} chars)")
            if retrieved_docs:
                logger.info(f"   ✅ Documents were used in generation")
            else:
                logger.warning(f"   ⚠️  No documents were used - only LLM knowledge")
            
            return "", history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Error in chat: {e}", exc_info=True)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        return []


def create_gradio_interface(chatbot: RAGChatbot) -> gr.Blocks:
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="RAG-based Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 RAG-based Chatbot
            
            Ask questions about your documents! The chatbot uses retrieval-augmented generation
            to provide accurate, context-aware responses based on your knowledge base.
            """
        )
        
        chatbot_interface = gr.Chatbot(
            label="Conversation",
            height=500,
            show_copy_button=True,
            type="messages"  # Use new messages format instead of deprecated tuples
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                scale=4,
                lines=2
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear History", variant="secondary")
        
        # Event handlers
        msg_input.submit(
            chatbot.chat,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface]
        )
        submit_btn.click(
            chatbot.chat,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface]
        )
        clear_btn.click(
            lambda: ([], ""),  # Clear history and input
            outputs=[chatbot_interface, msg_input]
        )
        
        gr.Markdown(
            """
            ### 💡 Tips
            - Ask specific questions for best results
            - The chatbot retrieves relevant context from your documents
            - Responses are generated using a local LLM via Ollama
            """
        )
    
    return demo


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Run the RAG-based chatbot with Gradio UI"
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='data/faiss_index.bin',
        help='Path to FAISS index file'
    )
    parser.add_argument(
        '--embed-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model name for vector store'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='llama2',
        help='Ollama model name (e.g., llama2, gemma:2b, mistral)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of retrieved documents to use as context'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='LLM temperature (0.0-1.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens in LLM response'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind the Gradio server'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to bind the Gradio server (use 0 to auto-find available port)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio share link'
    )
    
    args = parser.parse_args()
    
    # Initialize vector store
    logger.info(f"Loading vector store from {args.index_path}...")
    try:
        vector_store = FAISSVectorStore(
            index_path=args.index_path,
            embedding_model=args.embed_model
        )
        
        # Try to load existing index if it exists
        if os.path.exists(args.index_path):
            try:
                vector_store.load()
                num_docs = len(vector_store.texts)
                logger.info(f"Loaded existing FAISS index with {num_docs} documents")
                if num_docs == 0:
                    logger.error("Vector store is empty! Please rebuild embeddings using build_embeddings.py")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                logger.error("Please rebuild embeddings using build_embeddings.py")
                sys.exit(1)
        else:
            logger.error(
                f"Index file not found at {args.index_path}. "
                "Please run build_embeddings.py first to create the index."
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        sys.exit(1)
    
    # Initialize chatbot
    logger.info(f"Initializing RAG chatbot with model: {args.llm_model}")
    
    # Test Ollama connection before proceeding
    logger.info("Testing Ollama connection...")
    try:
        test_client = OllamaClient(model=args.llm_model)
        # Try a simple request to check if Ollama is running
        import requests
        test_url = f"{test_client.host}/api/tags"
        test_resp = requests.get(test_url, timeout=5)
        if test_resp.status_code == 200:
            logger.info("✅ Ollama is running and accessible")
            # Check if model is available
            models = test_resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if args.llm_model not in model_names:
                logger.warning(f"⚠️  Model '{args.llm_model}' not found in Ollama.")
                logger.warning(f"   Available models: {', '.join(model_names) if model_names else 'None'}")
                logger.warning(f"   Install it with: ollama pull {args.llm_model}")
        else:
            logger.warning(f"⚠️  Ollama responded with status {test_resp.status_code}")
    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to Ollama server!")
        logger.error("   Please make sure Ollama is running:")
        logger.error("   1. Start Ollama: ollama serve")
        logger.error("   2. Or check if it's running on a different port")
        logger.error("   3. Verify with: ollama list")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"⚠️  Could not verify Ollama connection: {e}")
        logger.warning("   Continuing anyway, but you may encounter errors...")
    
    chatbot = RAGChatbot(
        vector_store=vector_store,
        llm_model=args.llm_model,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Create and launch Gradio interface
    logger.info("Creating Gradio interface...")
    demo = create_gradio_interface(chatbot)
    
    # Handle port selection - check availability and find alternative if needed
    import socket
    
    def is_port_available(host: str, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return True
        except OSError:
            return False
    
    def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        for i in range(max_attempts):
            port = start_port + i
            if is_port_available(host, port):
                return port
        raise OSError(f"Could not find an available port after {max_attempts} attempts starting from {start_port}")
    
    port = args.port
    
    # If port is 0, auto-select any available port
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        logger.info(f"Auto-selected port: {port}")
    else:
        # Check if requested port is available
        if not is_port_available(args.host, port):
            logger.warning(f"Port {port} is already in use. Searching for alternative port...")
            try:
                port = find_available_port(args.host, port)
                logger.info(f"Using alternative port: {port}")
            except OSError as e:
                logger.error(f"Failed to find available port: {e}")
                logger.error(
                    f"Port {args.port} is busy. Please:\n"
                    f"  - Stop the process using port {args.port}, or\n"
                    f"  - Use --port 0 to auto-select, or\n"
                    f"  - Specify a different port with --port <number>"
                )
                sys.exit(1)
    
    logger.info(f"Launching Gradio server on {args.host}:{port}")
    try:
        demo.launch(
            server_name=args.host,
            server_port=port,
            share=args.share
        )
    except OSError as e:
        if "Cannot find empty port" in str(e) or "address already in use" in str(e).lower():
            logger.error(
                f"Failed to launch on port {port}. "
                f"Please try a different port with --port <number> or use --port 0 to auto-select."
            )
        raise


if __name__ == "__main__":
    main()

