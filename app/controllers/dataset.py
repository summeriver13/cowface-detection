import os

import yaml

from ..config import DATASET_CONFIG, DATASET_PATH


def create_dataset_yaml():
    """创建数据集配置文件"""
    yaml_path = os.path.join(DATASET_PATH, 'dataset.yaml')
    
    # 构建完整路径
    config = DATASET_CONFIG.copy()
    base_path = os.path.abspath(DATASET_PATH)
    
    # 检查并列出目录内容
    def check_directory(path, check_labels=True):
        if not os.path.exists(path):
            raise ValueError(f"Directory does not exist: {path}")
        
        images_dir = path
        labels_dir = path.replace('images', 'labels')
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if check_labels:
            label_files = [f.rsplit('.', 1)[0] + '.txt' for f in image_files]
            missing_labels = [f for f in label_files if not os.path.exists(os.path.join(labels_dir, f))]
            if missing_labels:
                print(f"Warning: Missing label files in {labels_dir}:")
                for f in missing_labels[:5]:
                    print(f"  - {f}")
                raise ValueError(f"Missing {len(missing_labels)} label files")
        
        print(f"Files in {path}:")
        for f in image_files[:5]:
            print(f"  - {f}")
        return len(image_files)
    
    # 验证并更新路径
    train_path = os.path.join(base_path, 'train', 'images')
    val_path = os.path.join(base_path, 'test', 'images')
    
    print("\nChecking directories...")
    train_files = check_directory(train_path)
    val_files = check_directory(val_path)
    
    print(f"\nFound {train_files} training images and {val_files} validation images")
    
    # 更新配置
    config.update({
        'path': base_path,
        'train': train_path,
        'val': val_path,
        'test': val_path,  # 使用验证集作为测试集
        'names': ['cow_face']  # 修改为列表格式
    })
    
    # 保存配置
    print("\nSaving dataset configuration...")
    print("Final config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"\nCreated dataset config at: {yaml_path}")
    
    return yaml_path
