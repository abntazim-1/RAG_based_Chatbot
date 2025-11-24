# chunker.py
"""
Recursive character text chunker for RAG pipelines.
- Splits documents using a hierarchy of separators (paragraphs, sentences, words, characters)
- Creates chunks with configurable size and overlap
- Simple, fast, and doesn't require ML models
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import nltk
from src.ingest.loader import DocumentLoader

# Download once (quiet)
nltk.download("punkt", quiet=True)


class SemanticChunker:
    """
    Recursive character text splitter that splits text using a hierarchy of separators.
    Tries to split by paragraphs, then sentences, then words, then characters.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,  # Kept for backward compatibility, not used
        similarity_threshold: Optional[float] = None,  # Kept for backward compatibility, not used
        max_chunk_tokens: int = 384,
        min_chunk_tokens: int = 50,
        chunk_overlap: int = 50,
        batch_size: Optional[int] = None,  # Kept for backward compatibility, not used
        device: Optional[str] = None,  # Kept for backward compatibility, not used
    ):
        """
        Args:
            max_chunk_tokens: Maximum number of words per chunk
            min_chunk_tokens: Minimum number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.max_tokens = max_chunk_tokens
        self.min_tokens = min_chunk_tokens
        self.chunk_overlap = chunk_overlap

    def _token_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator and clean up."""
        splits = text.split(separator)
        return [s.strip() for s in splits if s.strip()]

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Recursively split text using a hierarchy of separators.
        
        Args:
            text: Text to split
            separators: List of separators to try (in order of preference)
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap size in tokens
        
        Returns:
            List of text chunks
        """
        # If text is small enough, return as single chunk
        token_count = self._token_count(text)
        if token_count <= chunk_size:
            return [text] if token_count >= self.min_tokens else []

        # Try each separator in order
        for separator in separators:
            if separator in text:
                splits = self._split_text(text, separator)
                
                # If we got good splits, process them
                if len(splits) > 1:
                    # If splits are small enough, use them directly
                    if all(self._token_count(s) <= chunk_size for s in splits):
                        # Merge with overlap
                        return self._merge_chunks_with_overlap(splits, chunk_size, chunk_overlap)
                    else:
                        # Some splits are too large, recursively split them
                        chunks = []
                        for split in splits:
                            split_tokens = self._token_count(split)
                            if split_tokens <= chunk_size:
                                chunks.append(split)
                            else:
                                # Recursively split large chunks
                                sub_chunks = self._recursive_split(
                                    split, separators, chunk_size, chunk_overlap
                                )
                                chunks.extend(sub_chunks)
                        
                        # Merge all chunks with overlap
                        if chunks:
                            return self._merge_chunks_with_overlap(chunks, chunk_size, chunk_overlap)
        
        # If no separator worked, split by character count as last resort
        return self._split_by_characters(text, chunk_size, chunk_overlap)

    def _merge_chunks_with_overlap(
        self, chunks: List[str], chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Merge chunks ensuring they don't exceed chunk_size and have overlap."""
        if not chunks:
            return []
        
        merged_chunks = []
        current_chunk_words = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_words = chunk.split()
            chunk_tokens = len(chunk_words)
            
            # If single chunk is too large, split it first
            if chunk_tokens > chunk_size:
                # Save current chunk if exists
                if current_chunk_words:
                    merged = " ".join(current_chunk_words)
                    if self._token_count(merged) >= self.min_tokens:
                        merged_chunks.append(merged)
                    current_chunk_words = []
                    current_tokens = 0
                
                # Split the large chunk
                sub_chunks = self._split_by_characters(chunk, chunk_size, chunk_overlap)
                merged_chunks.extend(sub_chunks)
                continue
            
            # Check if adding this chunk would exceed size
            if current_tokens + chunk_tokens > chunk_size and current_chunk_words:
                # Save current chunk
                merged = " ".join(current_chunk_words)
                if self._token_count(merged) >= self.min_tokens:
                    merged_chunks.append(merged)
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_chunk_words) >= chunk_overlap:
                    overlap_words = current_chunk_words[-chunk_overlap:]
                    current_chunk_words = overlap_words + chunk_words
                    current_tokens = len(overlap_words) + chunk_tokens
                else:
                    current_chunk_words = chunk_words
                    current_tokens = chunk_tokens
            else:
                # Add to current chunk
                current_chunk_words.extend(chunk_words)
                current_tokens += chunk_tokens
        
        # Add final chunk (always include it, even if small)
        if current_chunk_words:
            merged = " ".join(current_chunk_words)
            # Include final chunk even if below min_tokens to avoid losing information
            if self._token_count(merged) >= self.min_tokens or not merged_chunks:
                merged_chunks.append(merged)
        
        return merged_chunks

    def _split_by_characters(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Split text by word count when other methods fail."""
        words = text.split()
        if not words:
            return []
        
        chunks = []
        i = 0
        
        while i < len(words):
            chunk_words = []
            chunk_tokens = 0
            
            # Build chunk up to chunk_size
            while i < len(words) and chunk_tokens < chunk_size:
                chunk_words.append(words[i])
                chunk_tokens += 1
                i += 1
            
            if chunk_words:
                chunk = " ".join(chunk_words)
                # Include chunk even if below min_tokens if it's the last one
                if self._token_count(chunk) >= self.min_tokens or i >= len(words):
                    chunks.append(chunk)
            
            # Move back for overlap
            if chunk_overlap > 0 and i < len(words):
                i = max(0, i - chunk_overlap)
        
        # If no chunks created but text exists, return at least one chunk
        if not chunks and text.strip():
            return [text] if self._token_count(text) >= self.min_tokens else [text]
        
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Define separator hierarchy: paragraphs, sentences, words
        separators = [
            "\n\n",  # Paragraphs (double newline)
            "\n",    # Single newline
            ". ",    # Sentences (period followed by space)
            "! ",    # Exclamation
            "? ",    # Question mark
            "; ",    # Semicolon
            ", ",    # Commas
            " ",     # Spaces (words)
        ]
        
        return self._recursive_split(
            text, separators, self.max_tokens, self.chunk_overlap
        )

    def chunk_documents(self, docs_dir: str) -> List[str]:
        """Chunk all .txt files in a directory."""
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
# ——————————————————————————————————————
# if __name__ == "__main__":
#     # Initialize chunker
#     chunker = SemanticChunker(
#         max_chunk_tokens=384,
#         min_chunk_tokens=50,
#         chunk_overlap=50,
#     )
#
#     # Chunk all .txt files in 'artifacts/'
#     chunks = chunker.chunk_documents("artifacts/")
#
#     print(f"Generated {len(chunks)} chunks")
#     for i, chunk in enumerate(chunks[:2]):
#         print(f"\n--- Chunk {i+1} ({chunker._token_count(chunk)} tokens) ---")
#         print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
