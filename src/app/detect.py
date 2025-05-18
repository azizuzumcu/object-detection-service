import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps
import io
from pathlib import Path

COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

session: ort.InferenceSession = None


def load_model(model_path: str = None):
    global session
    if session is None:
        base = Path(__file__).parent
        path = model_path or str(base / "model" / "yolo.onnx")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session


def letterbox(img: Image.Image, new_size: int = 640, color=(114, 114, 114)):
    """
    Returns:
      - padded+resized PIL Image,
      - resize ratio,
      - (pad_x, pad_y) tuple
    """
    w0, h0 = img.size
    r = new_size / max(w0, h0)
    new_unpad = (int(w0 * r), int(h0 * r))
    img_resized = img.resize(new_unpad, Image.BILINEAR)

    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    pad_x, pad_y = dw // 2, dh // 2
    img_padded = ImageOps.expand(
        img_resized, border=(pad_x, pad_y, dw - pad_x, dh - pad_y), fill=color
    )

    return img_padded, r, (pad_x, pad_y)


def preprocess(img_padded: Image.Image):
    arr = np.array(img_padded, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def xywh2xyxy(boxes: np.ndarray):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def postprocess(outputs, conf_thres: float = 0.25):
    pred = outputs[0][0]
    mask = pred[:, 4] > conf_thres
    pred = pred[mask]
    boxes_xywh = pred[:, :4]
    boxes = xywh2xyxy(boxes_xywh)
    scores = pred[:, 4]
    labels = np.argmax(pred[:, 5:], axis=1)
    return boxes, scores, labels


def non_max_suppression(boxes, scores, iou_threshold: float = 0.5):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def detect_objects(image_bytes: bytes, label: str = None):

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    img_padded, ratio, (pad_x, pad_y) = letterbox(img, new_size=640)
    inp = preprocess(img_padded)

    sess = load_model()
    outputs = sess.run(None, {"images": inp})

    boxes, scores, labels = postprocess(outputs, conf_thres=0.1)
    keep = non_max_suppression(boxes, scores, iou_threshold=0.3)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio

    results = []
    for box, score, cls in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        cls_name = COCO_LABELS[int(cls)]
        if label is None or cls_name == label:

            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, orig_w), min(y2, orig_h)
            results.append(
                {
                    "label": cls_name,
                    "confidence": float(score),
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return results
