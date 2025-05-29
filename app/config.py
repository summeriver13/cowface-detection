from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECRET_KEY = 'your-secret-key'  # 用于session加密

# 建议使用pymysql
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:123456@127.0.0.1/cow_face_db'
# SQLALCHEMY_DATABASE_URI = 'mysql://root:123456@127.0.0.1/cow_face_db'

SQLALCHEMY_TRACK_MODIFICATIONS = False

# 修改预训练模型配置
PRETRAINED_MODEL_PATH = {
    'n': PROJECT_ROOT / 'ml_models' / 'pretrained_models' / 'yolov8n.pt',
    's': PROJECT_ROOT / 'ml_models' / 'pretrained_models' / 'yolov8s.pt',
    'm': PROJECT_ROOT / 'ml_models' / 'pretrained_models' / 'yolov8m.pt',
    'l': PROJECT_ROOT / 'ml_models' / 'pretrained_models' / 'yolov8l.pt',
    'x': PROJECT_ROOT / 'ml_models' / 'pretrained_models' / 'yolov8x.pt'     # 添加 x 模型
}

# 在文件开头添加固定的模型路径

YOLO_MODEL_PATH = PROJECT_ROOT / 'ml_models' / 'yolo-models' / 'results-fine_tune_coco-weights-best.pt'
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / 'ml_models' / 'yolo-models' / 'cow_facial_recognition_efficientnet.pth'

# 修改训练配置
DATASET_PATH = PROJECT_ROOT / 'ml_models' / 'conference-paper-preparations'  # 修改检测数据集路径

DATASET_CONFIG = {
    'path': DATASET_PATH,
    'train': DATASET_PATH / 'train'/ 'images',  # 训练集图片路径
    'val': DATASET_PATH / 'test' / 'images',     # 验证集图片路径
    'test': DATASET_PATH / 'test' / 'images',    # 测试集图片路径
    'names': {
        0: 'cow_face'  # 类别名称
    },
    'nc': 1            # 类别数量
}

# cowFaceDetection-main/dataset_for_classification/train
TRAIN_DIR = PROJECT_ROOT / 'ml_models' / 'dataset_for_classification' / 'train'