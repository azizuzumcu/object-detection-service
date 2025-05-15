# src/app/main.py

from fastapi import FastAPI, File, UploadFile
import base64
from app.detect import detect_objects  # Eğer modul yolu farklıysa ".detect" olarak da deneyin

app = FastAPI()

@app.post("/detect")
@app.post("/detect/{label}")
async def detect(label: str = None, file: UploadFile = File(...)):
    # Gelen resmi oku
    image_bytes = await file.read()
    # Nesne tespiti yap
    results = detect_objects(image_bytes, label)
    # (Opsiyonel) Aynı resmi base64 olarak döndür
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "image": encoded_image,
        "objects": results,
        "count": len(results)
    }
