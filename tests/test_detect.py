import os
from app.detect import detect_objects

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_images")


def test_detect_car():
    img_path = os.path.join(TEST_DIR, "car.jpg")
    with open(img_path, "rb") as f:
        results = detect_objects(f.read(), label="car")
    assert any(obj["label"] == "car" for obj in results)
    assert len(results) >= 1


def test_detect_person():
    img_path = os.path.join(TEST_DIR, "person.jpg")
    with open(img_path, "rb") as f:
        results = detect_objects(f.read(), label="person")
    assert any(obj["label"] == "person" for obj in results)
    assert len(results) >= 1


def test_detect_multiple():
    img_path = os.path.join(TEST_DIR, "multi.jpg")
    with open(img_path, "rb") as f:
        results = detect_objects(f.read(), label=None)

    assert len(results) >= 2
