import base64
from pathlib import Path

from flask import Blueprint, request, session

from ..controllers import get_recognizer, require_login
from ..extensions import db
from ..models import Cattle, RecognitionRecord
from . import fail, ok

detect_api = Blueprint("detect", __name__)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOX_COLOR = (0, 255, 0)
LABEL_SCALE = 0.9
LABEL_THICKNESS = 2
LABEL_PADDING = 10


def _decode_upload(file):
    if Path(file.filename or "").suffix.lower() not in ALLOWED_SUFFIXES:
        return None

    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)


def _annotate(image, detections):
    import cv2

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['class']} ({detection['class_conf']})"
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, LABEL_THICKNESS)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICKNESS)
        cv2.rectangle(image, (x1, y1 - text_height - LABEL_PADDING), (x1 + text_width, y1), BOX_COLOR, -1)
        cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (0, 0, 0), LABEL_THICKNESS)
    return image


def _persist(detections) -> int:
    """识别结果按名字挂到同名牛只上；名字对不上就不入库，不编造默认牛只。"""
    cattle_by_name = {
        cattle.name: cattle for cattle in Cattle.query.filter_by(user_id=session["user_id"]).all()
    }
    saved = 0
    for detection in detections:
        cattle = cattle_by_name.get(detection["class"])
        if cattle is None:
            continue
        db.session.add(
            RecognitionRecord(
                cattle_id=cattle.cattle_id,
                result=(
                    f"检测置信度: {detection['detection_conf']}, "
                    f"识别置信度: {detection['class_conf']}, "
                    f"类别: {detection['class']}"
                ),
            )
        )
        saved += 1
    if saved:
        db.session.commit()
    return saved


@detect_api.post("/detect")
@require_login
def run_detection():
    if "image" not in request.files:
        return fail("missing_image", "没有上传图片")

    image = _decode_upload(request.files["image"])
    if image is None:
        return fail("unsupported_image", "只接受 jpg/png/bmp/webp 图片")

    try:
        detections = get_recognizer().detect_and_recognize(image)
    except Exception as exc:
        return fail("inference_failed", f"识别失败: {exc}", 500)

    import cv2

    saved = _persist(detections)
    encoded, buffer = cv2.imencode(".jpg", _annotate(image, detections))
    if not encoded:
        return fail("encode_failed", "结果图编码失败", 500)

    return ok(
        {
            "image": base64.b64encode(buffer).decode("utf-8"),
            "detections": detections,
            "saved_records": saved,
        }
    )
