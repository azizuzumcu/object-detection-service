# Temel imaj
FROM python:3.10-slim

# Sistem bağımlılıkları
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python bağımlılıkları
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
# src/ dizinindeki tüm dosyalar /app/ altında olacak şekilde
COPY src/ /app/

# Uvicorn ile FastAPI'yi başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
