from ..config import CLASSIFIER_MODEL_PATH, YOLO_MODEL_PATH
from ..extensions import db

import torch
from ultralytics import YOLO
import cv2
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

# 牛只个体信息表
class Cattle(db.Model):
    __tablename__ = 'cattle'
    cattle_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    records = db.relationship('RecognitionRecord', backref='cattle', lazy=True)

class CowFaceRecognizer:
    def __init__(self, train_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("使用设备:", self.device)
        
        # 加载YOLO模型和分类器
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.class_names = sorted(os.listdir(train_dir))  # 确保类别顺序一致
        self.classifier_model = self._load_classifier_model(CLASSIFIER_MODEL_PATH)
        
        print(f"可用的类别: {self.class_names}")  # 打印可用类别
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_classifier_model(self, model_path):
        num_classes = len(self.class_names)
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()

    def detect_and_recognize(self, image):
        try:
            # YOLO检测
            results = self.yolo_model(image)
            
            detections = []
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if conf < 0.5:  # 检测置信度阈值
                        continue
                        
                    # 裁剪牛脸区域并进行分类
                    face = img_pil.crop((x1, y1, x2, y2))
                    face_tensor = self.transform(face).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.classifier_model(face_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf_class, predicted = torch.max(probs, 1)
                        predicted_class = self.class_names[predicted.item()]
                        class_conf = float(conf_class[0])
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'detection_conf': f'{conf:.2f}',
                        'class': predicted_class,
                        'class_conf': f'{class_conf:.2f}'
                    })
                    
                    print(f"检测到牛脸: {predicted_class} (置信度: {class_conf:.2f})")
            
            return detections
            
        except Exception as e:
            print(f"检测和识别出错: {str(e)}")
            return []