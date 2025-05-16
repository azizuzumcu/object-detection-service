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

# Port metadata’sı (dokümantasyon için)
EXPOSE 8000

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY src/ .

# Başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
