import fitz  # PyMuPDF
from docx import Document


def clean_text(text: str) -> str:
    """
    Boş satırları ve gereksiz boşlukları temizler
    """
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines)


def load_pdf(path: str) -> str:
    """
    PDF'i sayfa sayfa okur
    """
    doc = fitz.open(path)
    text = ""

    for page in doc:
        text += page.get_text()

    return clean_text(text)


def load_docx(path: str) -> str:
    """
    DOCX'i paragraf paragraf okur
    """
    doc = Document(path)
    text = ""

    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"

    return clean_text(text)


def load_txt(path: str) -> str:
    """
    TXT dosyasını direkt okur
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return clean_text(text)


def load_file(path: str) -> str:
    """
    Dosya uzantısına göre uygun loader'ı seçer
    """
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".docx"):
        return load_docx(path)
    elif path.endswith(".txt"):
        return load_txt(path)
    else:
        raise ValueError("Desteklenmeyen format")