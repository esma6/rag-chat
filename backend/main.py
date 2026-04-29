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
from fastapi.responses import FileResponse, StreamingResponse
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
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):

    if state["index"] is None:
        raise HTTPException(400, "No documents uploaded")

    if req.active_files:
        active_metadata = [
            m for m in state["metadata"]
            if m["source"] in req.active_files
        ]
    else:
        active_metadata = state["metadata"]

    if len(active_metadata) == 0:
        def empty():
            yield "📭 Yüklenmiş aktif dosya yok."
        return StreamingResponse(empty(), media_type="text/plain")

    relevant = search_index(
        index=state["index"],
        query=req.message,
        metadata=active_metadata,
        k=5,
    )

    # Kaynak bilgisini bağlama dahil et
    context_parts = []
    for i, c in enumerate(relevant, 1):
        context_parts.append(
            f"[Kaynak {i} - {c['source']}]\n{c['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # PROFESYONEL SYSTEM PROMPT
    system_prompt = """Sen, kullanıcının yüklediği dokümanları analiz eden uzman bir asistansın.

GÖREVLERIN:
1. Sadece sana verilen BAĞLAM bilgisini kullanarak cevap ver
2. Bağlamda olmayan bilgileri ASLA uydurma
3. Eğer bağlamda cevap yoksa, "Bu bilgi yüklediğiniz dokümanlarda bulunmuyor" de
4. Cevaplarında hangi kaynaktan bilgi aldığını belirt (örn: "...[Kaynak 1]'e göre...")
5. Türkçe sorulara Türkçe, İngilizce sorulara İngilizce cevap ver
6. Cevaplarını yapılandırılmış ve okunabilir şekilde sun (gerekirse madde işaretleri kullan)
7. Özet istendiğinde ana noktaları vurgula
8. Sayısal veriler sorulduğunda kesin değerleri ver

YAPMAMAN GEREKENLER:
- Genel bilgini kullanma, sadece bağlamı kullan
- Tahminde bulunma
- Yanıltıcı veya eksik bilgi verme"""

    user_prompt = f"""BAĞLAM:
{context}

SORU: {req.message}

Yukarıdaki bağlamı kullanarak soruyu cevapla. Hangi kaynaktan bilgi aldığını belirtmeyi unutma."""

    def stream_generator():
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        response = state["groq_client"].chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            top_p=0.9,
            stream=True,
        )
        
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        
        if req.show_sources:
            sources = list({m["source"] for m in relevant})
            yield f"\n__SOURCES__:{','.join(sources)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")
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
# Docker'da /frontend, lokal'de ../frontend
DOCKER_FRONTEND = Path("/frontend")
LOCAL_FRONTEND = Path(__file__).parent.parent / "frontend"
FRONTEND_DIR = DOCKER_FRONTEND if DOCKER_FRONTEND.exists() else LOCAL_FRONTEND

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
@app.get("/preview")
def preview(filename: str):
    # 1. Dosyanın varlığını kontrol et
    # Backend'de dosyalar "uuid_filename.ext" formatında saklanıyor olabilir
    target_file = None
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.name.endswith(filename):
            target_file = file_path
            break

    if not target_file:
        print(f"Hata: {filename} dosyası bulunamadı. Mevcut dosyalar: {os.listdir(UPLOAD_DIR)}")
        raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {filename}")

    try:
        # 2. Dosyayı oku
        content = load_file(str(target_file))

        if not content or content.strip() == "":
            return {"content": "[Boş Doküman] Bu dosyanın içeriği boş veya metne çevrilemedi."}

        return {"content": content}
    except Exception as e:
        print(f"Okuma Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dosya okuma hatası: {str(e)}")
from fastapi.responses import FileResponse


# ------------------------------------------------------------------
# PDF GÖRÜNTÜLEME (IFRAME DESTEĞİ)
# ------------------------------------------------------------------
@app.get("/get_pdf/{filename}")
def get_pdf(filename: str):
    # Backend'deki uploads klasöründe ilgili dosyayı bulur
    target_file = None
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.name.endswith(filename):
            target_file = file_path
            break

    if not target_file:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")

    # Tarayıcının dosyayı indirmek yerine "görüntülemesini" sağlar
    return FileResponse(
        path=target_file,
        media_type='application/pdf',
        headers={"Content-Disposition": "inline"}  # Bu satır kritik!
    )

@app.get("/get_pdf/{filename}")
def get_pdf(filename: str):
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.name.endswith(filename):
            # Dosyayı tarayıcıda açılacak şekilde gönder
            return FileResponse(path=file_path, media_type='application/pdf')
    raise HTTPException(404, "Dosya bulunamadı")

@app.post("/remove_file")
def remove_file(filename: str):
    # 1. Bellekten (Metadata) temizle
    state["metadata"] = [m for m in state["metadata"] if m["source"] != filename]

    # 2. Aktif dosya listesinden çıkar
    if filename in state["files"]:
        del state["files"][filename]

    # 3. Diskteki fiziksel dosyayı sil
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.name.endswith(filename):
            try:
                os.remove(file_path)
            except:
                pass
    return {"status": "ok", "filename": filename}