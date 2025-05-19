# Object Detection Service

A lightweight, ONNX-based object detection API built with FastAPI.  
Upload an image, filter by COCO class (optional), and receive detected bounding boxes plus an annotated image (Base64).

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Requirements](#requirements)  
4. [Design Decisions](#design-decisions)  
5. [Model Conversion (YOLOv5 to ONNX)](#model-conversion-yolov5-to-onnx)  
6. [Installation & Running](#installation--running)  
7. [API Usage](#api-usage)  
8. [Testing](#testing)  
9. [License](#license)
---

## Overview

This service exposes a single REST endpoint that accepts an image upload and returns:  
- Detected objects (COCO labels + confidence + bounding boxes)  
- Total count of detected objects  
- The original image annotated with red boxes (Base64-encoded)  

It uses an ONNX YOLO model for inference, FastAPI for the HTTP API, and can be containerized with Docker.

---

## Features

- **ONNX Runtime** for fast, CPU-only inference  
- **FastAPI** for a simple, asynchronous REST interface  
- **Docker & Docker Compose** support for easy deployment  
- **Automated tests** (pytest), **linting** (flake8) and **formatting** (black) workflows  

---

## Requirements

- Python ≥ 3.10  
- (Optional) Docker & Docker Compose  

---

## Design Decisions & Assumptions

### Why FastAPI?
FastAPI was chosen as the web framework due to its asynchronous support, automatic OpenAPI documentation, and ease of integration with modern Python tooling. Its speed and developer experience make it ideal for building lightweight APIs like this one.

### Why ONNX?
The ONNX (Open Neural Network Exchange) format allows the model to be deployed in a platform-independent, optimized manner. It is especially useful for running inference on CPUs using the `onnxruntime`, which is lightweight and production-ready without requiring GPU support.

### Why Docker?
Docker provides a consistent, reproducible environment for the application. It ensures the service runs identically across machines, simplifies deployment, and makes integration with CI/CD pipelines seamless.

### Directory Structure
The `src/` folder contains the core application code, including:
- `app/`: the FastAPI logic and model logic
- `model/`: utilities to convert models to ONNX format
- `tests/`: unit tests using `pytest`
- `test_images/`: sample images for test cases

### Assumptions
- The inference will be performed on CPU environments.
- The user may or may not specify a target class label (`?label=car` is optional).
- All images are assumed to be valid JPEG/PNG files.
- Detected object labels follow COCO class names.
- The bounding boxes are returned as `[x1, y1, x2, y2]` format in image pixel space.

## Model Conversion: YOLOv5 to ONNX

This project uses a YOLOv5 model converted to ONNX format for efficient inference with `onnxruntime`.

### Why Convert to ONNX?

- **Framework Independence**: ONNX allows interoperability between different deep learning frameworks.
- **Performance**: Optimized for CPU inference using `onnxruntime`, which is lighter than PyTorch.
- **Deployment**: Easier to package and deploy in production, especially in Dockerized environments.

### Conversion Steps

The original YOLOv5 model (trained in PyTorch) can be converted to ONNX using the following command:

```bash
python export.py --weights yolov5s.pt --img 640 --batch 1 --device cpu --include onnx
```

This will generate a `yolov5s.onnx` file. In our project, the model is renamed and placed under:

```
src/app/model/yolo.onnx
```

### Notes

- We use `yolov5s` for speed and lightweight deployment. You can replace it with other YOLOv5 variants (e.g., `yolov5m`, `yolov5l`) if needed.
- Only the forward inference pass is included in ONNX; training-related ops are stripped.
- The exported ONNX model must match the input size expected by our preprocessing (e.g., 640x640).

## Installation & Running

You can run the project **locally with Python** or using **Docker**. Follow the method that suits you best.

---

### Option 1 — Run Locally (Python Environment)

#### 1. Clone the Repository

```bash
git clone https://github.com/azizuzumcu/object-detection-service.git
cd object-detection-service
```

#### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the API

```bash
cd src
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 5. Open in Browser

Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the API using Swagger UI.

---

### Option 2 — Run with Docker (Recommended for Production)

#### 1. Make Sure Docker & Docker Compose Are Installed

You can check:

```bash
docker -v
docker-compose -v
```

#### 2. Build and Run with Docker Compose

```bash
docker-compose up --build
```

#### 3. Access the API

Once the container is running, open:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

### Notes

- Your **YOLO ONNX model file** (`yolo.onnx`) should be located at `src/app/model/yolo.onnx`.  
- If the file doesn't exist, place a pretrained ONNX model there manually or include a download script.  
- You can test the API using Postman or via Swagger UI interface.


## API Usage

### POST /detect?label={optional}

**Request**:  
- Method: `POST`  
- Body: multipart/form-data  
- Field: `file` → image (jpg, png, etc.)  
- Query Param (optional): `label=person` or any COCO label  

**Response**:

```json
{
  "count": 2,
  "objects": [
    {
      "label": "car",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "annotated_image": "<base64 string>"
}
```

---

## Testing

This project includes automated tests for both the object detection logic and the FastAPI API endpoint.

### 1. Run All Tests

To execute all tests, activate your virtual environment and run:

```bash
pytest -q
```

If successful, you should see something like:

```
....                                                                [100%]
4 passed in 0.83s
```

### 2. Test Structure

The `tests/` directory contains:

```
tests/
├── test_detect.py     # Unit tests for the detect_objects() function
├── test_api.py        # Integration tests for the FastAPI endpoint
└── conftest.py        # Shared fixtures used across multiple test files
```

#### Example fixture (`conftest.py`):

```python
import pytest
from app.main import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)
```

This fixture allows all test files to use the `client` object for sending test HTTP requests.

### 3. Example Unit Test

```python
def test_detect_person():
    img_path = os.path.join(TEST_DIR, "person.jpg")
    with open(img_path, "rb") as f:
        results = detect_objects(f.read(), label="person")
    assert any(obj["label"] == "person" for obj in results)
```

### 4. Sample Test Images

Located in:

```
tests/test_images/
├── person.jpg
├── car.jpg
└── multi.jpg
```

Each image is used to test if the detection logic correctly identifies expected objects.

### 5. Test via cURL (Optional)

You can also manually test the API:

```bash
curl -X POST "http://localhost:8000/detect?label=car" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/test_images/car.jpg"
```

This will return a list of detected objects and a base64-encoded annotated image.

---

## License

This project is licensed under the **MIT License**.

### MIT License Summary

You are free to:

- **Use** the code for personal, academic, or commercial purposes  
- **Modify** it to suit your needs  
- **Distribute** it as part of your own projects  

Under the condition that you:

- **Include** the original copyright  
- **Include** the license text in any substantial portions of the software

### Full License Text

```
MIT License

Copyright (c) 2025 Aziz Üzümcü

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
