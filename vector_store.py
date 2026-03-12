import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def build_vector_store(chunks, index_dir="faiss_index"):
    os.makedirs(index_dir, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "docs.index"))

    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks, model


def load_vector_store(index_dir="faiss_index"):
    index = faiss.read_index(os.path.join(index_dir, "docs.index"))

    with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    return index, chunks, model


def retrieve_top_k(query, index, chunks, model, k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results