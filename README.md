# 🤖 Akademi AI - RAG Doküman Sohbet Uygulaması

Yüklediğiniz dokümanlarla yapay zeka destekli sohbet edebileceğiniz **RAG (Retrieval-Augmented Generation)** tabanlı bir uygulama. Llama 3.3 70B modeli ile dokümanlarınız üzerinden soru-cevap, özetleme ve bilgi çıkarımı yapabilirsiniz.

## ✨ Özellikler

- 📄 **Çoklu format desteği**: PDF, DOCX, DOC, TXT
- 🔍 **FAISS vektör araması** - Hızlı ve doğru bilgi çekimi
- 💬 **Streaming yanıt** - Gerçek zamanlı cevap üretimi
- 📚 **Kaynak gösterme** - Cevapların hangi dokümandan geldiği belirtilir
- 🎯 **Aktif/pasif doküman yönetimi** - Hangi dokümanların kullanılacağını seçin
- 🌓 **Açık/koyu tema** desteği
- 🐳 **Docker desteği** - Tek komutla çalıştırma
- ✅ **Birim testler** - 18 pytest testi ile

## 🛠️ Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------|
| Backend | FastAPI, Python 3.11 |
| LLM | Groq API (Llama 3.3 70B) |
| Embedding | SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Vektör DB | FAISS |
| Frontend | HTML, TailwindCSS, Vanilla JavaScript |
| Test | Pytest |
| Container | Docker, Docker Compose |

## 📋 Mimari (RAG Akışı)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Doküman    │────▶│ Metne Çevirme│────▶│   Chunking   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  FAISS Index │◀────│  Embedding   │◀────│    Chunks    │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       │ Kullanıcı Sorusu
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Benzerlik   │────▶│ İlgili Chunks│────▶│  Groq LLM    │
│   Araması    │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │ Cevap +      │
                                          │ Kaynaklar    │
                                          └──────────────┘
```

## 🚀 Kurulum

### Yöntem 1: Docker ile (Önerilen)

**Gereksinimler:** Docker Desktop

```bash
# Projeyi klonlayın
git clone https://github.com/esma6/rag-chat
cd rag-chat
```

`backend/.env` dosyası oluşturun:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

> 💡 Groq API anahtarı almak için: https://console.groq.com

Docker ile başlatın:
```bash
docker-compose up --build
```

Tarayıcıdan açın: **http://localhost:8000**

### Yöntem 2: Manuel Kurulum

**Gereksinimler:** Python 3.11+

```bash
git clone https://github.com/esma6/rag-chat
cd rag-chat/backend

# Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Bağımlılıkları yükle
pip install -r requirements.txt
```

`.env` dosyasını oluşturun (yukarıdaki gibi).

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Tarayıcıdan açın: **http://127.0.0.1:8000**

## 🎯 Kullanım

1. Sol panelden **"Yeni Doküman Yükle"** ile bir veya birden fazla doküman yükleyin
2. Yeşil noktaya tıklayarak dokümanları **aktif/pasif** yapabilirsiniz
3. Sohbet alanına sorunuzu yazın
4. Hızlı butonlarla **özet**, **kavramlar** veya **veri analizi** yaptırabilirsiniz
5. Cevaplarda hangi **kaynak**lardan bilgi alındığı belirtilir

## 🧪 Testler

Projeyi test etmek için:

```bash
cd rag-chat
pip install pytest pytest-asyncio httpx
pytest
```

**18 birim test** çalışacak: chunking, embedding, FAISS, document loader ve API testleri.

## 📡 API Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| POST | `/upload` | Doküman yükleme |
| POST | `/chat/stream` | Streaming sohbet |
| POST | `/remove_file` | Doküman silme |
| POST | `/toggle_file` | Aktif/pasif değiştir |
| GET | `/preview` | Doküman önizleme |
| GET | `/get_pdf/{filename}` | PDF görüntüleme |
| GET | `/health` | Sistem durumu |

## 📁 Proje Yapısı

```
rag-chat/
├── backend/
│   ├── main.py              # FastAPI uygulaması
│   ├── rag_engine.py        # RAG motoru (chunking, embedding, FAISS)
│   ├── document_loader.py   # Doküman okuma (PDF, DOCX, TXT)
│   ├── requirements.txt     # Python bağımlılıkları
│   ├── Dockerfile           # Docker yapılandırması
│   └── .env                 # API anahtarları (gitignore'da)
├── frontend/
│   └── index.html           # Tek sayfa uygulama
├── tests/
│   ├── test_rag_engine.py   # RAG motor testleri
│   ├── test_document_loader.py  # Doküman okuma testleri
│   ├── test_api.py          # API endpoint testleri
│   └── conftest.py          # Pytest fixtures
├── docker-compose.yml       # Docker Compose yapılandırması
├── pytest.ini               # Pytest yapılandırması
└── README.md
```

## 🔧 Çevre Değişkenleri

| Değişken | Açıklama | Varsayılan |
|----------|----------|------------|
| `GROQ_API_KEY` | Groq API anahtarı (zorunlu) | - |
| `GROQ_MODEL` | Kullanılacak Groq modeli | `llama-3.3-70b-versatile` |

## 📝 Lisans

MIT

## 👤 Geliştiriciler

[esma6](https://github.com/esma6) [ozgegul](https://github.com/ozgegul) [aysimaErgn](https://github.com/aysimaErgn)
