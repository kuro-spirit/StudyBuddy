import json
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer

# Paths (should match embed.py)
CHUNK_METADATA_PATH = "embeddings/chunk_metadata.json"
FAISS_INDEX_PATH = "embeddings/index.faiss"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model(model_name=EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def load_metadata() -> List[str]:
    with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    return np.array([model.encode(query)])

def retrieve_top_k(query: str, k: int = 5) -> List[str]:
    model = load_embedding_model()
    query_vec = embed_query(query, model)

    index = load_faiss_index()
    D, I = index.search(query_vec, k)

    all_chunks = load_metadata()
    results = [all_chunks[i] for i in I[0]]

    return results

if __name__ == "__main__":
    query = "what is a rank"
    results = retrieve_top_k(query, k = 3)

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(results):
        print(f"\n[{i+1}] {chunk}\n")