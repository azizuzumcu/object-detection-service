FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

EXPOSE 8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
