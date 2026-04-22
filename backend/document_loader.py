import fitz  # PyMuPDF
from docx import Document
import subprocess
import os


def clean_text(text: str) -> str:
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines)


# ---------------- PDF ----------------
def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text()

    return clean_text(text)


# ---------------- DOCX ----------------
def load_docx(path: str) -> str:
    doc = Document(path)
    text = ""

    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"

    return clean_text(text)


# ---------------- DOC (eski Word) ----------------
def load_doc(path: str) -> str:
    """
    DOC dosyasını antiword ile okur (en stabil yöntem)
    """
    try:
        result = subprocess.run(
            ["antiword", path],
            capture_output=True,
            text=True,
            check=True
        )
        return clean_text(result.stdout)

    except FileNotFoundError:
        raise RuntimeError(
            "antiword kurulu değil. Linux: sudo apt install antiword"
        )

    except Exception as e:
        raise RuntimeError(f"DOC okuma hatası: {str(e)}")


# ---------------- TXT ----------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


# ---------------- ROUTER ----------------
def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext == ".doc":
        return load_doc(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError("Desteklenmeyen format")