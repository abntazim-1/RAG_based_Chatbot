import os
import time
import logging
from typing import List, Optional, Union
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sentence_transformers import SentenceTransformer
# 
from src.utils.exception import CustomException
from src.ingest.chunker import SemanticChunker
from src.retriever.vector_store import VectorStore, FAISSVectorStore
import logging
logging.basicConfig(level=logging.INFO)

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 16, chunker: Optional[SemanticChunker]=None, vector_store: Optional[VectorStore]=None):
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logging.error(f"Could not load embedding model {model_name}: {e}")
            raise CustomException(e, sys)
        # Use SemanticChunker directly
        self.chunker = chunker if chunker else SemanticChunker()
        # Create FAISSVectorStore with defaults if no vector_store provided
        if vector_store is None:
            # Default index path and use the same embedding model
            default_index_path = "data/faiss_index.bin"
            # Ensure data directory exists
            os.makedirs(os.path.dirname(default_index_path), exist_ok=True)
            self.vector_store = FAISSVectorStore(index_path=default_index_path, embedding_model=model_name)
        else:
            self.vector_store = vector_store
        self.logger = logging.getLogger(self.__class__.__name__)
        # Number of texts to pass to model.encode in a single call (prevents large-list memory blowups)
        self.texts_per_encode = 512

    def embed_text(self, chunks: List[str], store: bool = False, meta: Optional[dict] = None) -> List[np.ndarray]:
        try:
            start = time.time()
            all_embeddings: List[np.ndarray] = []
            total = len(chunks)
            for start_idx in range(0, total, self.texts_per_encode):
                end_idx = min(start_idx + self.texts_per_encode, total)
                batch_chunks = chunks[start_idx:end_idx]
                batch_embeddings_np = self.model.encode(
                    batch_chunks,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                batch_embeddings: List[np.ndarray] = [emb for emb in batch_embeddings_np]
                if store:
                    self.store_embeddings(batch_embeddings, batch_chunks, meta, base_index=start_idx)
                all_embeddings.extend(batch_embeddings)
            elapsed = time.time() - start
            self.logger.info(
                f"Embedded {len(chunks)} chunks in {elapsed:.2f} seconds using model '{self.model_name}' (batched)."
            )
            return all_embeddings
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise CustomException(e, sys)

    def embed_file(self, file_path: str, store: bool = False) -> List[np.ndarray]:
        try:
            # Read file and chunk via SemanticChunker
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            chunks = self.chunker.chunk_text(text)
            if not chunks:
                self.logger.warning(f"No text chunks generated from file {file_path}.")
                return []
            meta = {"source": os.path.basename(file_path)}
            return self.embed_text(chunks, store=store, meta=meta)
        except Exception as e:
            self.logger.error(f"Error embedding file {file_path}: {e}")
            raise CustomException(e, sys)

    def store_embeddings(self, embeddings: List[np.ndarray], chunks: List[str], meta: Optional[dict] = None, base_index: int = 0):
        try:
            chunk_metas = []
            ids = []
            for idx, chunk in enumerate(chunks):
                cm = {"chunk_id": idx, "text": chunk}
                if meta:
                    cm.update(meta)
                chunk_metas.append(cm)
                source = meta.get('source', 'file') if meta else 'file'
                ids.append(f"chunk_{source}_{base_index + idx}")
            self.vector_store.add_embeddings(embeddings, chunk_metas, ids)
            self.logger.info(f"Stored {len(embeddings)} embeddings in vector store.")
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {e}")
            raise CustomException(e, sys)
