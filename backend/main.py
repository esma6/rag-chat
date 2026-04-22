"""
RAG Chat - ACTIVE / PASSIVE FIXED VERSION
"""

import os
import uuid
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_loader import load_file
from rag_engine import chunk_text, embed_texts, create_index, search_index

load_dotenv()

# ------------------------------------------------------------------
# APP
# ------------------------------------------------------------------
app = FastAPI(title="RAG Chat Active/Passive FIX", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc"}

state = {
    "index": None,
    "metadata": [],
    "groq_client": None,

    # filename -> True/False
    "files": {}
}

# ------------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------------
@app.on_event("startup")
def startup():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY eksik")

    state["groq_client"] = Groq(api_key=api_key)

# ------------------------------------------------------------------
# MODELS
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    show_sources: bool = True
    active_files: list[str] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
def generate_answer(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([c["content"] for c in context_chunks])

    prompt = f"""
Aşağıdaki bağlamı kullanarak cevap ver.

BAĞLAM:
{context}

SORU:
{question}

CEVAP:
"""

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    res = state["groq_client"].chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )

    return res.choices[0].message.content.strip()

# ------------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------------
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": "unsupported format"
            })
            continue

        save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"

        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        file.file.close()

        try:
            text = load_file(str(save_path))
            chunks, metadata = chunk_text(text, filename=file.filename)
            embeddings = embed_texts(chunks)

            # ------------------------
            # INDEX UPDATE
            # ------------------------
            if state["index"] is None:
                state["index"] = create_index(embeddings)
            else:
                state["index"].add(embeddings)

            # ------------------------
            # METADATA FIX
            # ------------------------
            for m in metadata:
                m["source"] = file.filename

            state["metadata"].extend(metadata)

            # default ACTIVE
            state["files"][file.filename] = True

            results.append({
                "filename": file.filename,
                "status": "ok",
                "chunks": len(chunks)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": str(e)
            })

    return {"results": results}

# ------------------------------------------------------------------
# TOGGLE ACTIVE / PASSIVE
# ------------------------------------------------------------------
@app.post("/toggle_file")
def toggle_file(filename: str):

    if filename not in state["files"]:
        return {"error": "file not found"}

    state["files"][filename] = not state["files"][filename]

    return {
        "filename": filename,
        "active": state["files"][filename]
    }

# ------------------------------------------------------------------
# CHAT (🔥 FINAL FIXED LOGIC)
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if state["index"] is None:
        raise HTTPException(400, "No documents uploaded")

    # 🔥 AKTİF DOSYA FİLTRE
    if req.active_files:
        active_metadata = [
            m for m in state["metadata"]
            if m["source"] in req.active_files
        ]
    else:
        active_metadata = state["metadata"]

    # ❌ BOŞSA DUR
    if len(active_metadata) == 0:
        return ChatResponse(
            answer="📭 Yüklenmiş aktif dosya yok.",
            sources=[]
        )

    # 🔥 BURASI EKSİK OLAN KISIMDI
    relevant = search_index(
        index=state["index"],
        query=req.message,
        metadata=active_metadata,
        k=5,
    )

    answer = generate_answer(req.message, relevant)

    sources = list({r["source"] for r in relevant}) if req.show_sources else []

    return ChatResponse(
        answer=answer,
        sources=sources
    )

# ------------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "chunks": len(state["metadata"]),
        "files": state["files"]
    }

# ------------------------------------------------------------------
# FRONTEND
# ------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)