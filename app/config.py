import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_MODELS_DIR = Path(os.environ.get("ML_MODELS_DIR", PROJECT_ROOT / "ml_models"))

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-key")

SQLALCHEMY_DATABASE_URI = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{(PROJECT_ROOT / 'instance' / 'cow_face.db').as_posix()}",
)
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 上传图片上限，超过直接 413，避免整包读进内存
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# 检测与识别权重
YOLO_MODEL_PATH = ML_MODELS_DIR / "yolo-models" / "results-fine_tune_coco-weights-best.pt"
CLASSIFIER_MODEL_PATH = ML_MODELS_DIR / "yolo-models" / "cow_facial_recognition_efficientnet.pth"
CLASSIFIER_CLASSES_PATH = CLASSIFIER_MODEL_PATH.with_suffix(".classes.json")
TRAIN_DIR = ML_MODELS_DIR / "dataset_for_classification" / "train"

# 训练用的 YOLOv8 预训练权重，按模型规格取
PRETRAINED_MODEL_PATH = {
    size: ML_MODELS_DIR / "pretrained_models" / f"yolov8{size}.pt" for size in "nsmlx"
}

DATASET_PATH = ML_MODELS_DIR / "conference-paper-preparations"
DATASET_CONFIG = {
    "path": DATASET_PATH,
    "train": DATASET_PATH / "train" / "images",
    "val": DATASET_PATH / "test" / "images",
    "test": DATASET_PATH / "test" / "images",
    "names": {0: "cow_face"},
    "nc": 1,
}
