# Dockerfile
FROM python:3.10-slim

# Sistemi güncelle, gerekli paketleri ekle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements’i önce kopyala ve yükle (cache optimumu için)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uygulamanın kaynak kodunu kopyala
COPY src/ /app/src/

# UVicorn ile çalıştır
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
