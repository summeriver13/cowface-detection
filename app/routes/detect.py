import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request, session

from ..controllers import get_available_models, get_recognizer, require_login
from ..extensions import db
from ..models import Cattle, RecognitionRecord

detect_bp = Blueprint('detect-routes', __name__)

ALLOWED_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@detect_bp.route('/detect_page')
@require_login
def detect_page():
    return render_template('detect.html', models=get_available_models())


@detect_bp.route('/detect', methods=['POST'])
@require_login
def detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '没有上传图片'}), 400

    file = request.files['image']
    if Path(file.filename or '').suffix.lower() not in ALLOWED_SUFFIXES:
        return jsonify({'success': False, 'error': '图片格式不支持'}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'success': False, 'error': '图片格式不支持'}), 400

    try:
        detections = get_recognizer().detect_and_recognize(img)
    except Exception as exc:
        return jsonify({'success': False, 'error': f'识别失败: {exc}'}), 500

    cattle = Cattle.query.filter_by(user_id=session['user_id']).first()
    if cattle is None:
        cattle = Cattle(name='默认牛只', user_id=session['user_id'])
        db.session.add(cattle)
        db.session.flush()

    for detection in detections:
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
    db.session.commit()

    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = f"{detection['class']} ({detection['class_conf']})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        )
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    encoded, buffer = cv2.imencode('.jpg', img)
    if not encoded:
        return jsonify({'success': False, 'error': '结果图编码失败'}), 500

    return jsonify(
        {
            'success': True,
            'image': base64.b64encode(buffer).decode('utf-8'),
            'detections': detections,
        }
    )
