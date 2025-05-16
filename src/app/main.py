# src/app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from app.detect import detect_objects

app = FastAPI(title="Yolo ONNX Object Detection")

@app.post("/detect", summary="Upload image and detect objects")
async def detect_endpoint(
    label: str = Query(None, description="Filter by this label, e.g. car"),
    file: UploadFile = File(..., description="JPEG/PNG image to analyze")
):
    # Content-type kontrolü
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Only JPEG or PNG files are supported")
    # Bytes’a çevir
    image_bytes = await file.read()
    # Algoritmayı çağır
    results = detect_objects(image_bytes, label=label)
    # Hata durum kontrolü
    if not results:
        return JSONResponse(status_code=200, content={"message": "No objects found", "objects": []})
    return {"objects": results, "count": len(results)}
