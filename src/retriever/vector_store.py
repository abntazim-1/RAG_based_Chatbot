"""
vector_store.py
----------------
Handles embedding storage and semantic search for the RAG system.
This module defines a base VectorStore interface and an example
FAISSVectorStore implementation for local retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer


# -----------------------------
# Base Class for Vector Storage
# -----------------------------
class VectorStore(ABC):
    """
    Abstract base class for vector storage implementations.
    """

    @abstractmethod
    def add(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add new text embeddings and metadata to the vector store."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents for the query."""
        pass


# -----------------------------
# FAISS Implementation
# -----------------------------
class FAISSVectorStore(VectorStore):
    """
    Local FAISS-based vector store implementation.
    Efficient for small to mid-sized RAG systems.
    """

    def __init__(self, index_path: str, embedding_model: str):
        """
        Args:
            index_path (str): Path to store/load FAISS index.
            embedding_model (str): Name of SentenceTransformer model.
        """
        self.index_path = index_path
        self.texts_path = index_path.replace('.bin', '_texts.json') if index_path.endswith('.bin') else f"{index_path}_texts.json"
        self.metadatas_path = index_path.replace('.bin', '_metadatas.json') if index_path.endswith('.bin') else f"{index_path}_metadatas.json"
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Add new documents to the FAISS index.
        """
        if not texts:
            return

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in texts])
    
    def add_embeddings(self, embeddings: List[np.ndarray], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """
        Add pre-computed embeddings to the FAISS index.
        
        Args:
            embeddings: List of numpy arrays (pre-computed embeddings)
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (not used in FAISS but kept for compatibility)
        """
        if not embeddings:
            return
        
        # Convert list of numpy arrays to a single numpy array
        embeddings_array = np.array([emb if isinstance(emb, np.ndarray) else np.array(emb) for emb in embeddings], dtype=np.float32)
        self.index.add(embeddings_array)
        
        # Extract texts from metadatas if available
        texts = []
        for meta in (metadatas or []):
            if isinstance(meta, dict) and 'text' in meta:
                texts.append(meta['text'])
            else:
                texts.append("")  # Placeholder if text not in metadata
        
        # If texts weren't in metadata, we need them separately
        # For now, we'll expect them in metadata
        if not texts or all(not t for t in texts):
            raise ValueError("Texts must be provided in metadatas with 'text' key when using add_embeddings()")
        
        self.texts.extend(texts)
        self.metadatas.extend(metadatas or [{} for _ in embeddings])

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query.
        """
        if not self.index:
            import logging
            logging.warning("FAISS index is not initialized")
            return []
        
        if len(self.texts) == 0:
            import logging
            logging.warning(f"Vector store has no texts loaded! Index has {self.index.ntotal} vectors but texts list is empty.")
            logging.warning("This usually means the index was saved without texts. Please rebuild using build_embeddings.py")
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            # Handle invalid indices (can happen if index.ntotal > len(texts))
            if idx < 0 or idx >= len(self.texts):
                import logging
                logging.warning(f"Invalid index {idx} (texts length: {len(self.texts)})")
                continue
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx] if idx < len(self.metadatas) else {},
                "score": float(score)
            })
        return results

    def save(self):
        """Save FAISS index, texts, and metadatas to disk."""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save texts as JSON
        with open(self.texts_path, 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)
        
        # Save metadatas as JSON
        with open(self.metadatas_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def load(self):
        """Load FAISS index, texts, and metadatas from disk."""
        # Load FAISS index
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        # Load texts
        if os.path.exists(self.texts_path):
            with open(self.texts_path, 'r', encoding='utf-8') as f:
                self.texts = json.load(f)
        else:
            raise FileNotFoundError(f"Texts file not found: {self.texts_path}. Please rebuild the index.")
        
        # Load metadatas
        if os.path.exists(self.metadatas_path):
            with open(self.metadatas_path, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
        else:
            # If metadatas file doesn't exist, create empty metadatas for each text
            self.metadatas = [{} for _ in self.texts]
        
        # Verify consistency
        if len(self.texts) != self.index.ntotal:
            raise ValueError(
                f"Mismatch: index has {self.index.ntotal} vectors but {len(self.texts)} texts. "
                "Please rebuild the index."
            )


# -----------------------------
# Example Usage (for testing)
# -----------------------------
if __name__ == "__main__":
    store = FAISSVectorStore(index_path="data/faiss_index.bin",
                             embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    # Example documents
    texts = [
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language models.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Ollama can run lightweight language models locally on consumer hardware."
    ]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]

    store.add(texts, metadatas)
    results = store.search("What is RAG?", top_k=2)

    for r in results:
        print(f"\nScore: {r['score']:.4f}\nText: {r['text']}\nMetadata: {r['metadata']}")
