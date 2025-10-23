
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(text_chunks):
    vectors = embedder.encode(text_chunks)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))
    return index

def search(index, query, text_chunks, k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
