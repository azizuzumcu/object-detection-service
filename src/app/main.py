from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app.detect import detect_objects
from PIL import Image, ImageDraw, ImageFont
import io
import base64

app = FastAPI(title="Object Detection Service")

@app.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    label: str = Query(None, description="Filtrelemek isterseniz COCO etiketi, örn: car")
):
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz resim dosyası")

    detections = detect_objects(content, label=label)

    draw = ImageDraw.Draw(img)
    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f'{obj["label"]} {obj["confidence"]:.2f}'
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except:
            font = ImageFont.load_default()
        # metni ayarlamak için textbbox kullanıyoruz
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((text_bbox[0], text_bbox[1]), text, fill="white", font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    
    payload = {
        "objects": detections,
        "count": len(detections),
        "annotated_image": f"data:image/jpeg;base64,{encoded}"
    }

    # jsonable_encoder ile içerde kalan numpy/scalar objeleri dönüştürüyoruz
    return JSONResponse(content=jsonable_encoder(payload))
