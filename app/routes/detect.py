from flask import Blueprint, jsonify, render_template, request, redirect, session, url_for
from ..config import YOLO_MODEL_PATH, CLASSIFIER_MODEL_PATH
from ..controllers import require_login
from ..controllers import get_available_models, load_yolo_model
from ..models import Cattle, RecognitionRecord, CowFaceRecognizer
from ..extensions import db

from ultralytics import YOLO
import base64
import cv2
import numpy as np

from datetime import datetime

detect_bp = Blueprint('detect-routes', __name__)

@detect_bp.route('/detect_page')
@require_login
def detect_page():
    models = get_available_models()
    return render_template('detect.html', models=models)

@detect_bp.route('/detect', methods=['POST'])
@require_login
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传图片'})
        
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': '图片格式不支持'})
        
        # 检测和识别
        from ..config import TRAIN_DIR
        
        load_yolo_model()
        recognizer = CowFaceRecognizer(train_dir=TRAIN_DIR)
        detections = recognizer.detect_and_recognize(img)
        
        # 保存识别记录
        
        # 获取用户的第一头牛（或创建一个新的）
        cattle = Cattle.query.filter_by(user_id=session['user_id']).first()
        if not cattle:
            cattle = Cattle(
                name="默认牛只",
                user_id=session['user_id']
            )
            db.session.add(cattle)
            db.session.commit()
        
        # 为每个检测结果创建记录
        for det in detections:
            try:
                record = RecognitionRecord(
                    cattle_id=cattle.cattle_id,
                    timestamp=datetime.utcnow(),
                    result=f"检测置信度: {det['detection_conf']}, 识别置信度: {det['class_conf']}, 类别: {det['class']}"
                )
                db.session.add(record)
                print(f"添加识别记录: {record.result}")
            except Exception as e:
                print(f"保存记录时出错: {str(e)}")
                continue
        
        try:
            db.session.commit()
            print("成功保存所有记录")
        except Exception as e:
            print(f"提交记录时出错: {str(e)}")
            db.session.rollback()
            
        # 在图片上绘制结果
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class']} ({det['class_conf']})"
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(img, (x1, y1 - 30), (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # 添加标签文字
            cv2.putText(img, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'detections': detections
        })
        
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'处理出错: {str(e)}'
        }), 500