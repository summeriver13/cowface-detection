from ..config import PRETRAINED_MODEL_PATH
from ..extensions import db
from ..models import ModelTraining

from ultralytics import YOLO
from datetime import datetime
import os
import torch
import multiprocessing

def train_model(app, model_type, epochs, batch_size, learning_rate, user_id):
    with app.app_context():
        training = ModelTraining(
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            status='training',
            user_id=user_id
        )
        db.session.add(training)
        db.session.commit()
        
        try:
            # 创建数据集配置文件
            from . import create_dataset_yaml
            dataset_yaml = create_dataset_yaml()
            
            # 获取预训练模型
            pretrained_path = PRETRAINED_MODEL_PATH[model_type]
            print(f"Using pretrained model: {pretrained_path}")
            
            if not os.path.exists(pretrained_path):
                model = YOLO(f'yolov8{model_type}.pt')
                os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
                model.save(pretrained_path)
            else:
                model = YOLO(pretrained_path)
                        
            # 开始训练
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                imgsz=640,
                save=True,
                project='runs/train',
                name=f'fine_tune_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                exist_ok=True,
                pretrained=True,
                resume=False,
                patience=50,
                workers=0,
                single_cls=True,
                rect=False,
                cache=False,
                amp=False,
                seed=42,
                verbose=True
            )
            
            print("Training completed successfully")
            
            # 更新训练结果
            training.end_time = datetime.utcnow()
            
            if hasattr(results, 'maps'):
                training.map50 = results.maps.get(50, None)
                # 这里假设results有map50_95字段或者你需要自己计算，先用getattr避免异常
                training.map50_95 = getattr(results, 'map50_95', None)
                training.precision = getattr(results, 'precision', None)
                training.recall = getattr(results, 'recall', None)
            training.status = 'completed'
            training.description = f'Training completed. Results saved in {results.save_dir}'
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            training.status = 'failed'
            training.description = str(e)
            raise

        finally:
            db.session.commit()