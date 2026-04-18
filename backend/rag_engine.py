import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. MODEL HAZIRLIĞI
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. CHUNKING (METİN BÖLME)
def chunk_text(text, filename, chunk_size=600, overlap=100):
    chunks = []
    metadata = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        metadata.append({
            "source": filename,
            "content": chunk
        })

        start += (chunk_size - overlap)

    return chunks, metadata

# 3. EMBEDDING ÜRETİMİ
def embed_texts(texts):
    embeddings = model.encode(texts)
    return np.array(embeddings).astype('float32')

# 4. FAISS İNDEKS YÖNETİMİ
def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# 5. RETRIEVAL (ARAMA)
def search_index(index, query, metadata, k=5):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    return [metadata[i] for i in indices[0]]