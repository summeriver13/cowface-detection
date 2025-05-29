from ..config import PRETRAINED_MODEL_PATH
from ..extensions import db
from ..models import ModelTraining

from ultralytics import YOLO
from datetime import datetime
import os

def train_model(app, model_type, epochs, batch_size, learning_rate, user_id):
    print(f"Starting training with: model={model_type}, epochs={epochs}, batch={batch_size}, lr={learning_rate}")
    
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
            from ..controllers import create_dataset_yaml
            dataset_yaml = create_dataset_yaml()
            
            # 获取预训练模型
            pretrained_path = PRETRAINED_MODEL_PATH[model_type]
            print(f"Using pretrained model: {pretrained_path}")
            
            if not os.path.exists(pretrained_path):
                print(f"Downloading YOLOv8{model_type} model...")
                model = YOLO(f'yolov8{model_type}.pt')
                os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
                model.save(pretrained_path)
            
            model = YOLO(pretrained_path)
            print("Model loaded successfully")
            
            # 开始训练
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                device='cpu',
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
                training.map50 = results.maps[50]
                training.map50_95 = results.maps[50:95].mean()
                training.precision = results.precision
                training.recall = results.recall
            training.status = 'completed'
            training.description = f'Training completed. Results saved in {results.save_dir}'
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            training.status = 'failed'
            training.description = str(e)
            raise
        finally:
            db.session.commit()