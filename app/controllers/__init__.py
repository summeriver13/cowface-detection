from .auth import require_login
from .dataset import create_dataset_yaml
from .yolov8_models import get_available_models
from .train import train_model

import os

# 添加环境变量来禁用安全检查
os.environ['TORCH_SAFE_LOADING'] = '1'
os.environ['ULTRALYTICS_SKIP_SAFETY_CHECK'] = '1'