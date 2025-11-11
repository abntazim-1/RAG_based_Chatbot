"""
retriever.py
------------
Handles document retrieval logic for the RAG system.
It fetches the most relevant context chunks from the vector store
based on a user's query embedding.
"""

from typing import List, Dict, Any
from .vector_store import VectorStore


class Retriever:
    """
    Retriever handles fetching of relevant documents
    from the vector store using semantic similarity.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        """
        Initialize the Retriever.
        Args:
            vector_store (VectorStore): Instance of a vector store.
            top_k (int): Number of top documents to retrieve.
        """
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a given query.

        Args:
            query (str): User's natural language question.

        Returns:
            List[Dict[str, Any]]: Retrieved documents sorted by relevance.
        """
        if not query or not query.strip():
            return []

        # Get the top relevant documents from the vector store
        results = self.vector_store.search(query, top_k=self.top_k)
        return results


# ---------------------------
# Example Usage (for testing)
# ---------------------------
if __name__ == "__main__":
    from .vector_store import FAISSVectorStore  # Example concrete implementation

    # Example setup (mock usage)
    store = FAISSVectorStore(index_path="data/index.faiss", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(store, top_k=5)

    query = "What is the purpose of retrieval augmented generation?"
    docs = retriever.retrieve(query)

    print("\nTop Retrieved Chunks:\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.get('text')[:150]}...\n")
