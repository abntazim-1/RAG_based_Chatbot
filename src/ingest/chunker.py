# chunker.py
"""
Semantic-aware document chunker for RAG pipelines.
- Loads local 'all-mpnet-base-v2' embedding model (offline)
- Splits documents into coherent chunks using sentence-level similarity
- Optimized for CPU, batch inference, and low memory
- Integrates with LlamaIndex / LangChain via list of strings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Iterable

import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.ingest.loader import DocumentLoader 
# Download once (quiet)
nltk.download("punkt", quiet=True)


class SemanticChunker:
    """
    High-performance semantic chunker using local embedding model.
    """

    def __init__(
        self,
        model_path: str = "./models/all-mpnet-base-v2",
        similarity_threshold: float = 0.75,
        max_chunk_tokens: int = 384,
        min_chunk_tokens: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = 384
        self.sim_threshold = similarity_threshold
        self.max_tokens = max_chunk_tokens
        self.min_tokens = min_chunk_tokens
        self.batch_size = batch_size

    def _token_count(self, text: str) -> int:
        return len(text.split())

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

    def _embed_batch(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _merge_by_similarity(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[str]:
        if not sentences:
            return []

        chunks: List[str] = []
        current: List[str] = [sentences[0]]
        current_tokens = self._token_count(sentences[0])

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            next_tokens = self._token_count(sentences[i])
            would_exceed = current_tokens + next_tokens > self.max_tokens

            if sim < self.sim_threshold or would_exceed:
                chunk = " ".join(current)
                if self._token_count(chunk) >= self.min_tokens:
                    chunks.append(chunk)
                current = [sentences[i]]
                current_tokens = next_tokens
            else:
                current.append(sentences[i])
                current_tokens += next_tokens

        # Final chunk
        final = " ".join(current)
        if self._token_count(final) >= self.min_tokens:
            chunks.append(final)

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if self._token_count(text) >= self.min_tokens else []

        embeddings = self._embed_batch(sentences)
        return self._merge_by_similarity(sentences, embeddings)

    def chunk_documents(self, docs_dir: str) -> List[str]:
        path = Path(docs_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {docs_dir}")

        all_chunks: List[str] = []
        txt_files = list(path.glob("*.txt"))

        if not txt_files:
            raise ValueError(f"No .txt files found in {docs_dir}")

        for file in txt_files:
            text = file.read_text(encoding="utf-8")
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)

        return all_chunks


# ——————————————————————————————————————
# Usage Example
# # ——————————————————————————————————————
# if __name__ == "__main__":
#     # Initialize chunker with local model
#     chunker = SemanticChunker(
#         model_path="./models/all-mpnet-base-v2",
#         similarity_threshold=0.75,
#         max_chunk_tokens=384,
#         device="cpu",
#     )

#     # Chunk all .txt files in 'artifacts/'
#     chunks = chunker.chunk_documents("artifacts/")

#     print(f"Generated {len(chunks)} semantic chunks")
#     for i, chunk in enumerate(chunks[:2]):
#         print(f"\n--- Chunk {i+1} ({chunker._token_count(chunk)} tokens) ---")
#         print(chunk[:300] + ("..." if len(chunk) > 300 else ""))