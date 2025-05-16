# src/app/main.py

from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.detect import detect_objects
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = FastAPI(title="Object Detection Service")


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]


class DetectResponse(BaseModel):
    objects: List[Detection]
    count: int
    annotated_image: str  # data:image/jpeg;base64,...


@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    label: Optional[str] = Query(None, description="Filtrelemek isterseniz COCO etiketi, örn: car")
):
    # Gelen dosyayı okuyalım
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz resim dosyası")

    # Nesne tespiti
    detections = detect_objects(content, label=label)

    # Görüntü üzerine çizim
    draw = ImageDraw.Draw(img)
    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f'{obj["label"]} {obj["confidence"]:.2f}'

        # font ve metin boyutu
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except:
            font = ImageFont.load_default()
        # Pillow ≥8.0: use textbbox, eski sürüm için getbbox/getsize alternatif
        try:
            text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
        except AttributeError:
            text_w, text_h = font.getmask(text).size

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
        draw.text((x1, y1 - text_h), text, fill="white", font=font)

    # İşaretlenmiş resmi Base64 e çevir
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{annotated_b64}"

    return JSONResponse({
        "objects": detections,
        "count": len(detections),
        "annotated_image": data_uri
    })
