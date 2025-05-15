# src/app/detect.py

import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import io
from pathlib import Path

# COCO’nun 80 sınıf adı
COCO_LABELS = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant",
    "bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

session: ort.InferenceSession = None

def load_model(model_path: str = None):
    global session
    if session is None:
        base = Path(__file__).parent
        path = model_path or str(base / "model" / "yolo.onnx")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session

def letterbox(img: Image.Image, new_size: int = 640, color=(114,114,114)):
    w0, h0 = img.size
    r = new_size / max(w0, h0)
    new_unpad = (int(w0 * r), int(h0 * r))
    img = img.resize(new_unpad, Image.BILINEAR)
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    left, top = dw // 2, dh // 2
    right, bottom = dw - left, dh - top
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=color)

def preprocess(image_bytes: bytes, img_size: int = 640):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = letterbox(img, new_size=img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1x3xHxW
    return arr

def postprocess(outputs, conf_thres: float = 0.25):
    pred = outputs[0][0]  # (N,85)
    mask = pred[:, 4] > conf_thres
    pred = pred[mask]
    boxes = pred[:, :4].tolist()
    scores = pred[:, 4].tolist()
    labels = np.argmax(pred[:, 5:], axis=1).tolist()
    return boxes, scores, labels

def non_max_suppression(boxes, scores, iou_threshold: float = 0.5):
    if not boxes:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def detect_objects(image_bytes: bytes, label: str = None):
    sess = load_model()
    img_in = preprocess(image_bytes, img_size=640)
    outputs = sess.run(None, {"images": img_in})

    # Eşik sonrası
    boxes, scores, labels = postprocess(outputs, conf_thres=0.1)

    # NMS ile çoğulları temizle
    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    labels = [labels[i] for i in keep]

    results = []
    for box, score, cls_idx in zip(boxes, scores, labels):
        cls_name = COCO_LABELS[cls_idx]
        if label is None or cls_name == label:
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            results.append({
                "label": cls_name,
                "confidence": float(score),
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
            })
    return results
