# Python 3.11 slim base image
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Pip'i güncelle
RUN pip install --no-cache-dir --upgrade pip

# Önce torch'un CPU sürümünü kur (GPU/CUDA paketlerini önlemek için)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Sonra requirements'ı kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend kodunu kopyala
COPY . .

# Uploads klasörünü oluştur
RUN mkdir -p uploads

# Port aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]