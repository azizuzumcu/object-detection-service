import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def white_jpeg():
    from PIL import Image
    img = Image.new("RGB", (20, 20), color=(255,255,255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_detect_endpoint_basic(white_jpeg):
    files = {"file": ("white.jpg", white_jpeg, "image/jpeg")}
    resp = client.post("/detect?label=car", files=files)
    assert resp.status_code == 200, resp.text
    j = resp.json()
    assert "count" in j and "objects" in j and "annotated_image" in j
    assert isinstance(j["objects"], list)
