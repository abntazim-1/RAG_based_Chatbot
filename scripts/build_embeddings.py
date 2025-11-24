"""
build_embeddings.py
-------------------
Build embeddings from documents in the Artifacts/ directory.

This script:
1. Loads all documents from Artifacts/
2. Chunks them using SemanticChunker
3. Embeds the chunks using Embedder
4. Stores everything in FAISSVectorStore
5. Saves the index, texts, and metadatas to disk
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest.loader import DocumentLoader
from src.ingest.chunker import SemanticChunker
from src.retriever.vector_store import FAISSVectorStore

# Try to import logger, fallback to standard logging if it fails
try:
    from src.utils.logger import logger
except (ImportError, AttributeError):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("BuildEmbeddings")


async def build_embeddings(
    artifacts_dir: str = "artifacts",
    index_path: str = "data/faiss_index.bin",
    embed_model: str = "./models/all-mpnet-base-v2",
    chunker_model_path: str = "./models/all-mpnet-base-v2",
    similarity_threshold: float = 0.75,
    max_chunk_tokens: int = 384,
    min_chunk_tokens: int = 20,  # Reduced to prevent losing small but important chunks
    embed_batch_size: int = 16,
    device: str = "cpu"
):
    """
    Build embeddings from documents in Artifacts directory.
    
    Args:
        artifacts_dir: Directory containing documents
        index_path: Path to save FAISS index
        embed_model: Embedding model name for vector store
        chunker_model_path: Path to semantic chunker model
        similarity_threshold: Similarity threshold for chunking
        max_chunk_tokens: Maximum tokens per chunk
        min_chunk_tokens: Minimum tokens per chunk
        embed_batch_size: Batch size for embedding
        device: Device for models (cpu/cuda)
    """
    
    # Check if artifacts directory exists
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        logger.error(f"Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)
    
    # Get all files from artifacts directory
    all_files = []
    for ext in ['*.pdf', '*.txt', '*.md', '*.docx', '*.pptx', '*.csv', '*.html']:
        all_files.extend(list(artifacts_path.glob(ext)))
        all_files.extend(list(artifacts_path.glob(ext.upper())))
    
    if not all_files:
        logger.error(f"No supported documents found in {artifacts_dir}")
        logger.info("Supported formats: .pdf, .txt, .md, .docx, .pptx, .csv, .html")
        sys.exit(1)
    
    logger.info(f"Found {len(all_files)} document(s) in {artifacts_dir}")
    for f in all_files:
        logger.info(f"  - {f.name}")
    
    # Step 1: Load documents
    logger.info("\n" + "="*60)
    logger.info("Step 1: Loading documents...")
    logger.info("="*60)
    
    loader = DocumentLoader()
    file_paths = [str(f) for f in all_files]
    documents = await loader.load(file_paths)
    
    logger.info(f"✅ Loaded {len(documents)} document(s)")
    total_chars = sum(len(doc.content) for doc in documents)
    logger.info(f"   Total content: {total_chars:,} characters")
    
    if not documents:
        logger.error("No documents were loaded. Exiting.")
        sys.exit(1)
    
    # Step 2: Chunk documents using SemanticChunker
    logger.info("\n" + "="*60)
    logger.info("Step 2: Chunking documents with SemanticChunker...")
    logger.info("="*60)
    
    try:
        chunker = SemanticChunker(
            model_path=chunker_model_path,
            similarity_threshold=similarity_threshold,
            max_chunk_tokens=max_chunk_tokens,
            min_chunk_tokens=min_chunk_tokens,
            device=device
        )
    except FileNotFoundError:
        logger.error(f"Chunker model not found at {chunker_model_path}")
        logger.info("Please ensure the model is downloaded or specify a different path")
        sys.exit(1)
    
    all_chunks = []
    chunk_metadata = []
    
    total_input_chars = 0
    total_output_chars = 0
    
    for doc in documents:
        doc_chars = len(doc.content)
        total_input_chars += doc_chars
        chunks = chunker.chunk_text(doc.content)
        
        if not chunks:
            logger.warning(f"⚠️  No chunks generated for document: {doc.source}")
            logger.warning(f"   Document length: {doc_chars} characters, {chunker._token_count(doc.content)} tokens")
            continue
        
        all_chunks.extend(chunks)
        chunk_chars = sum(len(c) for c in chunks)
        total_output_chars += chunk_chars
        
        # Create metadata for each chunk
        for i, chunk in enumerate(chunks):
            chunk_metadata.append({
                "source": doc.source,
                "chunk_num": i,
                "total_chunks": len(chunks),
                **doc.metadata
            })
        
        logger.info(f"   📄 {Path(doc.source).name}: {len(chunks)} chunks, {doc_chars:,} → {chunk_chars:,} chars")
    
    logger.info(f"✅ Generated {len(all_chunks)} chunks from {len(documents)} document(s)")
    if all_chunks:
        avg_chunk_len = sum(len(c) for c in all_chunks) / len(all_chunks)
        avg_chunk_tokens = sum(chunker._token_count(c) for c in all_chunks) / len(all_chunks)
        logger.info(f"   Average chunk length: {avg_chunk_len:.0f} characters, {avg_chunk_tokens:.1f} tokens")
        logger.info(f"   Total input: {total_input_chars:,} chars → Output: {total_output_chars:,} chars")
        logger.info(f"   Coverage: {(total_output_chars/total_input_chars*100):.1f}%")
        logger.info(f"   Sample chunk (first 150 chars): {all_chunks[0][:150]}...")
    
    if not all_chunks:
        logger.error("No chunks were generated. Exiting.")
        sys.exit(1)
    
    # Validate chunk sizes
    too_small = [i for i, c in enumerate(all_chunks) if chunker._token_count(c) < min_chunk_tokens]
    too_large = [i for i, c in enumerate(all_chunks) if chunker._token_count(c) > max_chunk_tokens * 1.5]
    
    if too_small:
        logger.warning(f"⚠️  {len(too_small)} chunks are below min_tokens ({min_chunk_tokens})")
    if too_large:
        logger.warning(f"⚠️  {len(too_large)} chunks exceed max_tokens by 50% (>{max_chunk_tokens * 1.5})")
    
    # Step 3: Initialize vector store
    logger.info("\n" + "="*60)
    logger.info("Step 3: Initializing vector store...")
    logger.info("="*60)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(index_path) if os.path.dirname(index_path) else "."
    os.makedirs(data_dir, exist_ok=True)
    
    vector_store = FAISSVectorStore(
        index_path=index_path,
        embedding_model=embed_model
    )
    
    # Step 4: Embed and store chunks
    logger.info(f"Embedding and storing {len(all_chunks)} chunks using model '{embed_model}'...")
    logger.info("This may take a few minutes...")
    
    # Prepare texts and metadatas for vector store
    texts = all_chunks  # Use the chunks directly as texts
    metadatas = []
    
    # Add chunk index to metadata for tracking
    for i, meta in enumerate(chunk_metadata):
        meta["chunk_index"] = i
        metadatas.append(meta)
    
    # Validate before storing
    if len(texts) != len(metadatas):
        logger.error(f"❌ Mismatch: {len(texts)} texts but {len(metadatas)} metadatas")
        sys.exit(1)
    
    # Use add() which will compute embeddings and store everything
    logger.info(f"Computing embeddings for {len(texts)} chunks...")
    vector_store.add(texts, metadatas)
    
    # Verify storage
    stored_count = len(vector_store.texts)
    index_count = vector_store.index.ntotal
    
    if stored_count != len(texts):
        logger.error(f"❌ Storage mismatch: Expected {len(texts)} chunks, got {stored_count}")
        sys.exit(1)
    
    if index_count != len(texts):
        logger.error(f"❌ Index mismatch: Expected {len(texts)} vectors, got {index_count}")
        sys.exit(1)
    
    logger.info(f"✅ Stored {stored_count} chunks in vector store")
    logger.info(f"   Vector store has {index_count} vectors")
    logger.info(f"   All chunks successfully stored and indexed")
    
    # Step 4: Save everything to disk
    logger.info("\n" + "="*60)
    logger.info("Step 4: Saving vector store to disk...")
    logger.info("="*60)
    
    vector_store.save()
    
    logger.info(f"✅ Saved FAISS index to: {index_path}")
    logger.info(f"✅ Saved texts to: {vector_store.texts_path}")
    logger.info(f"✅ Saved metadatas to: {vector_store.metadatas_path}")
    
    # Verify the save
    logger.info("\n" + "="*60)
    logger.info("Verification:")
    logger.info("="*60)
    logger.info(f"✅ Index file exists: {os.path.exists(index_path)}")
    logger.info(f"✅ Texts file exists: {os.path.exists(vector_store.texts_path)}")
    logger.info(f"✅ Metadatas file exists: {os.path.exists(vector_store.metadatas_path)}")
    logger.info(f"✅ Total documents: {len(vector_store.texts)}")
    logger.info(f"✅ Total vectors: {vector_store.index.ntotal}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Embedding build complete!")
    logger.info("="*60)
    logger.info(f"You can now run the chatbot with:")
    logger.info(f"  python scripts/run_app.py --index-path {index_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build embeddings from documents in Artifacts/ directory"
    )
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        default='artifacts',
        help='Directory containing documents (default: artifacts)'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='data/faiss_index.bin',
        help='Path to save FAISS index (default: data/faiss_index.bin)'
    )
    parser.add_argument(
        '--embed-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--chunker-model',
        type=str,
        default='./models/all-mpnet-base-v2',
        help='Path to semantic chunker model (default: ./models/all-mpnet-base-v2)'
    )
    parser.add_argument(
        '--sim-threshold',
        type=float,
        default=0.75,
        help='Similarity threshold for chunking (default: 0.75)'
    )
    parser.add_argument(
        '--max-chunk-tokens',
        type=int,
        default=384,
        help='Maximum tokens per chunk (default: 384)'
    )
    parser.add_argument(
        '--min-chunk-tokens',
        type=int,
        default=20,
        help='Minimum tokens per chunk (default: 20)'
    )
    parser.add_argument(
        '--embed-batch',
        type=int,
        default=16,
        help='Batch size for embedding (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for models: cpu or cuda (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(build_embeddings(
        artifacts_dir=args.artifacts_dir,
        index_path=args.index_path,
        embed_model=args.embed_model,
        chunker_model_path=args.chunker_model,
        similarity_threshold=args.sim_threshold,
        max_chunk_tokens=args.max_chunk_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        embed_batch_size=args.embed_batch,
        device=args.device
    ))


if __name__ == "__main__":
    main()

