import os
import faiss
import numpy as np
import json
from typing import List
from sentence_transformers import SentenceTransformer
from ingest import ingest

# You can switch this model later if needed
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_METADATA_PATH = "embeddings/chunk_metadata.json"
FAISS_INDEX_PATH = "embeddings/index.faiss"


def load_embedding_model(model_name=EMBED_MODEL_NAME):
    print(f"[INFO] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_chunks(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    print(f"[INFO] Embedding {len(chunks)} chunks...")
    return np.array(model.encode(chunks, show_progress_bar=True))


def save_metadata(chunks: List[str]):
    """Save the raw chunks as metadata (for retrieval reference)."""
    os.makedirs("embeddings", exist_ok=True)
    with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved chunk metadata to {CHUNK_METADATA_PATH}")


def build_faiss_index(embeddings: np.ndarray):
    print(f"[INFO] Building FAISS index with dim={embeddings.shape[1]}")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[INFO] FAISS index saved to {FAISS_INDEX_PATH}")


def process_and_index(chunks: List[str]):
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model)
    save_metadata(chunks)
    build_faiss_index(embeddings)

if __name__ == "__main__":
    file_path = "data/notes.pdf"
    chunks = ingest(file_path)
    process_and_index(chunks)