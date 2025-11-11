import argparse
import asyncio
from typing import List

import os
import sys
# Ensure project root is on sys.path so 'src.*' imports work when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest.loader import DocumentLoader, Document
from src.ingest.chunker import SemanticChunker
from src.ingest.embedder import Embedder


async def load_docs(paths: List[str]) -> List[Document]:
    loader = DocumentLoader()
    return await loader.load(paths)


async def main():
    parser = argparse.ArgumentParser(description="Demo: loader -> semantic chunker -> embedder pipeline")
    parser.add_argument('paths', nargs='+', help='Files or directories to load')
    parser.add_argument('--model-path', type=str, default="./models/all-mpnet-base-v2", help='Local model dir for SemanticChunker')
    parser.add_argument('--sim-threshold', type=float, default=0.75, help='Sentence adjacency similarity threshold')
    parser.add_argument('--max-chunk-tokens', type=int, default=384, help='Max words per chunk (approx)')
    parser.add_argument('--min-chunk-tokens', type=int, default=50, help='Minimum words per chunk')
    parser.add_argument('--device', type=str, default='cpu', help='Device for SemanticChunker model')
    parser.add_argument('--embed-model', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformers embedding model name')
    parser.add_argument('--embed-batch', type=int, default=16, help='Embedding batch size')
    parser.add_argument('--store', action='store_true', help='Store embeddings in the configured VectorStore')
    args = parser.parse_args()

    # 1) Load
    docs = await load_docs(args.paths)
    print(f"Loaded {len(docs)} documents.")
    if not docs:
        return

    # 2) Chunk
    chunker = SemanticChunker(
        model_path=args.model_path,
        similarity_threshold=args.sim_threshold,
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        device=args.device,
    )

    all_chunks: List[str] = []
    for d in docs:
        chunks = chunker.chunk_text(d.content)
        all_chunks.extend(chunks)
    print(f"Generated {len(all_chunks)} chunks.")

    # Show a preview
    for i, ch in enumerate(all_chunks[:2]):
        preview = (ch[:200] + "...") if len(ch) > 200 else ch
        print(f"\n--- Chunk {i+1} preview ---\n{preview}")

    # 3) Embed (+ optional store)
    if all_chunks:
        embedder = Embedder(model_name=args.embed_model, batch_size=args.embed_batch)
        # Attach minimal source meta if only one directory/file was used
        meta = {"source": args.paths[0]}
        _ = embedder.embed_text(all_chunks, store=args.store, meta=meta)
        print(f"Embedded {len(all_chunks)} chunks. Stored: {bool(args.store)}")


if __name__ == "__main__":
    asyncio.run(main())


