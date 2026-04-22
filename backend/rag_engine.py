import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ================================
# MODEL
# ================================
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ================================
# METADATA DOSYA
# ================================
METADATA_FILE = "metadata.json"

def save_metadata(metadata):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ================================
# CHUNKING
# ================================
def chunk_text(text, filename, chunk_size=600, overlap=100):
    chunks = []
    metadata = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        metadata.append({
            "doc_id": filename,
            "source": filename,
            "content": chunk,
            "active": True,
            "deleted": False
        })

        start += (chunk_size - overlap)

    return chunks, metadata

# ================================
# EMBEDDING
# ================================
def embed_texts(texts):
    embeddings = model.encode(texts)
    return np.array(embeddings).astype('float32')

# ================================
# FAISS INDEX
# ================================
def create_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ================================
# SEARCH (FİLTRELİ RAG)
# ================================
def search_index(index, query, metadata, k=5, selected_doc=None):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k * 5)

    results = []

    for i in indices[0]:
        if i >= len(metadata):
            continue

        meta = metadata[i]

        if meta.get("deleted"):
            continue

        if selected_doc:
            if meta["doc_id"] != selected_doc:
                continue
        else:
            if not meta.get("active", True):
                continue

        results.append(meta)

        if len(results) >= k:
            break

    return results

# ================================
# DOKÜMAN YÖNETİMİ
# ================================
def set_active(metadata, doc_id, active_status):
    for m in metadata:
        if m["doc_id"] == doc_id:
            m["active"] = active_status

def delete_doc(metadata, doc_id):
    for m in metadata:
        if m["doc_id"] == doc_id:
            m["deleted"] = True

def list_docs(metadata):
    docs = {}
    for m in metadata:
        doc_id = m["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "active": m.get("active", True),
                "deleted": m.get("deleted", False)
            }
    return docs