import json
from functools import lru_cache
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

from ..config import CLASSIFIER_CLASSES_PATH, CLASSIFIER_MODEL_PATH, TRAIN_DIR, YOLO_MODEL_PATH

DETECTION_CONFIDENCE_THRESHOLD = 0.5
CLASSIFIER_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_class_names(train_dir: Path) -> list[str]:
    """类别顺序来自训练时的目录名，首次加载快照到 json，之后不再随目录变化漂移。"""
    if CLASSIFIER_CLASSES_PATH.exists():
        return json.loads(CLASSIFIER_CLASSES_PATH.read_text(encoding="utf-8"))

    names = sorted(entry.name for entry in train_dir.iterdir() if entry.is_dir())
    if not names:
        raise FileNotFoundError(f"分类数据集目录为空: {train_dir}")

    CLASSIFIER_CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_CLASSES_PATH.write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return names


class CowFaceRecognizer:
    def __init__(self, train_dir: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(str(YOLO_MODEL_PATH))
        self.class_names = _load_class_names(Path(train_dir))
        self.classifier_model = self._load_classifier_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _load_classifier_model(self) -> nn.Module:
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.class_names)),
        )
        state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

    def detect_and_recognize(self, image) -> list[dict]:
        results = self.yolo_model(image)
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
                face = img_pil.crop((x1, y1, x2, y2))
                face_tensor = self.transform(face).unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    probabilities = torch.softmax(self.classifier_model(face_tensor), dim=1)
                    class_confidence, predicted = torch.max(probabilities, 1)

                detections.append(
                    {
                        'box': [x1, y1, x2, y2],
                        'detection_conf': f'{confidence:.2f}',
                        'class': self.class_names[predicted.item()],
                        'class_conf': f'{float(class_confidence[0]):.2f}',
                    }
                )

        return detections


@lru_cache(maxsize=1)
def get_recognizer() -> CowFaceRecognizer:
    """权重几百 MB，进程内只加载一次，首次请求付出代价。"""
    return CowFaceRecognizer(TRAIN_DIR)
