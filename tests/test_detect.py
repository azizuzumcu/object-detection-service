# tests/test_detect.py
import io
import pytest
from PIL import Image
from app.detect import detect_objects

@pytest.fixture
def tiny_car_image():
    # Çok küçük (20×20) transparent imaj yapıp kaydediyoruz
    img = Image.new("RGB", (20, 20), color=(255,255,255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_detect_returns_list(tiny_car_image):
    # label=None ile çağırınca mutlaka liste dönmeli
    results = detect_objects(tiny_car_image, label=None)
    assert isinstance(results, list)
    # boş da olsa, count integer olmalı
    # burada hiç obje bulamaması normal
    assert all("label" in r and "confidence" in r and "bbox" in r for r in results)
