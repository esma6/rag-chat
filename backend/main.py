"""
RAG Chat - main.py
YZTA 5.0 P2P2 Challenge: "Kendi Dokümanların ile Sohbet Et"
"""

import os
import uuid
import shutil
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from document_loader import load_file
from rag_engine import chunk_text, embed_texts, create_index, search_index

load_dotenv()

# ---------------------------------------------------------------------------
# Uygulama & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Chat API",
    description="Kendi Dokümanların ile Sohbet Et — YZTA 5.0 P2P2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Upload dizini
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc"}

# ---------------------------------------------------------------------------
# Uygulama durumu
# ---------------------------------------------------------------------------
state = {
    "index": None,
    "metadata": [],
    "groq_client": None,   # startup'ta initialize edilecek
}

# ---------------------------------------------------------------------------
# Startup: Groq client burada oluşturuluyor
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(".env dosyasında GROQ_API_KEY tanımlı değil!")
    state["groq_client"] = Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Pydantic şemaları
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    show_sources: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# ---------------------------------------------------------------------------
# Yardımcı: Groq ile cevap üret
# ---------------------------------------------------------------------------
def generate_answer(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([c["content"] for c in context_chunks])
    prompt = f"""Aşağıdaki bağlam bilgilerini kullanarak soruyu Türkçe olarak cevapla.
Sadece bağlamda geçen bilgilere dayan. Bağlamda yoksa "Bu bilgiye sahip değilim." de.

BAĞLAM:
{context}

SORU: {question}

CEVAP:"""

    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    response = state["groq_client"].chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# ENDPOINT — Sağlık kontrolü
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    sources = list({m["source"] for m in state["metadata"]})
    return {
        "status": "ok",
        "indexed_chunks": len(state["metadata"]),
        "documents": sources,
    }

# ---------------------------------------------------------------------------
# ENDPOINT — Döküman yükleme
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": f"Desteklenmeyen format: {suffix}",
            })
            continue

        save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
        try:
            with save_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})
            continue
        finally:
            file.file.close()

        try:
            text = load_file(str(save_path))
        except Exception as e:
            save_path.unlink(missing_ok=True)
            results.append({"filename": file.filename, "status": "error", "detail": f"Okuma hatası: {e}"})
            continue

        try:
            chunks, metadata = chunk_text(text, filename=file.filename)
            embeddings = embed_texts(chunks)

            if state["index"] is None:
                state["index"] = create_index(embeddings)
            else:
                state["index"].add(embeddings)

            state["metadata"].extend(metadata)

            results.append({
                "filename": file.filename,
                "status": "ok",
                "chunks": len(chunks),
            })
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "detail": f"İndeksleme hatası: {e}"})

    return {"results": results}

# ---------------------------------------------------------------------------
# ENDPOINT — Sohbet
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if state["index"] is None or len(state["metadata"]) == 0:
        raise HTTPException(
            status_code=400,
            detail="Henüz hiç döküman yüklenmedi. Lütfen önce bir dosya yükleyin.",
        )

    relevant = search_index(
        index=state["index"],
        query=req.message,
        metadata=state["metadata"],
        k=5,
    )

    answer = generate_answer(question=req.message, context_chunks=relevant)
    sources = list({chunk["source"] for chunk in relevant}) if req.show_sources else []

    return ChatResponse(answer=answer, sources=sources)

# ---------------------------------------------------------------------------
# Frontend'i serv et (opsiyonel)
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)