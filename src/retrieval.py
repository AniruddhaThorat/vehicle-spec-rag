import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-mpnet-base-v2")

def retrieve(query, chunks, index, top_k=5):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in I[0]]