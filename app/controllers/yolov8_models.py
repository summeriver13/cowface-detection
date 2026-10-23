from ..config import PRETRAINED_MODEL_PATH

from ultralytics import YOLO
import glob
import os


def get_available_models():
    """获取所有可用的模型"""
    models = {
        'pretrained': {
            # 列出所有YOLOv8模型，无论是否下载
            'n': {'name': 'YOLOv8n (纳米版)', 'path': PRETRAINED_MODEL_PATH['n'], 'available': False},
            's': {'name': 'YOLOv8s (小型版)', 'path': PRETRAINED_MODEL_PATH['s'], 'available': False},
            'm': {'name': 'YOLOv8m (中型版)', 'path': PRETRAINED_MODEL_PATH['m'], 'available': False},
            'l': {'name': 'YOLOv8l (大型版)', 'path': PRETRAINED_MODEL_PATH['l'], 'available': False},
            'x': {'name': 'YOLOv8x (超大版)', 'path': PRETRAINED_MODEL_PATH['x'], 'available': False}
        },
        'trained': []  # 训练好的模型
    }

    for model_type in models['pretrained']:
        if os.path.exists(PRETRAINED_MODEL_PATH[model_type]):
            models['pretrained'][model_type]['available'] = True

    if os.path.exists('runs/train'):
        for model_dir in glob.glob('runs/train/*/weights/best.pt'):
            name = model_dir.split(os.sep)[-3]  # 获取训练实验名称
            models['trained'].append({
                'name': name,
                'path': model_dir,
                'available': True
            })

    # 一个可用模型都没有时下载最小的那个，避免检测页是空的
    if not any(m['available'] for m in models['pretrained'].values()) and not models['trained']:
        try:
            os.makedirs('ml_models/pretrained_models', exist_ok=True)
            print("Downloading YOLOv8n model...")
            model = YOLO('yolov8n.pt')
            os.replace(model.ckpt_path, PRETRAINED_MODEL_PATH['n'])
            models['pretrained']['n']['available'] = True
        except Exception as e:
            print(f"Error downloading model: {str(e)}")

    return models
