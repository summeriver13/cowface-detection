def train_model(model_type, epochs, batch_size, learning_rate, user_id, dataset_path):
    global is_training, stop_training
    is_training = True
    stop_training = False
    
    try:
        # 记录训练开始
        training = ModelTraining(
            user_id=user_id,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            status='training'
        )
        db.session.add(training)
        db.session.commit()

        # 创建数据集配置
        yaml_path = create_dataset_yaml(dataset_path)
        
        # 加载预训练模型
        model = YOLO(PRETRAINED_MODEL_PATH[model_type])
        
        # 开始训练
        try:
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                lr0=learning_rate,
                device=0 if torch.cuda.is_available() else 'cpu',
                callbacks=[StopTrainingCallback()]  # 添加回调以检查停止标志
            )
            
            # 如果训练被中止
            if stop_training:
                training.status = 'cancelled'
                training.description = '用户手动停止训练'
            else:
                # 训练正常完成
                training.status = 'completed'
                training.map50 = float(results.results_dict.get('metrics/mAP50(B)', 0))
                training.map50_95 = float(results.results_dict.get('metrics/mAP50-95(B)', 0))
                training.precision = float(results.results_dict.get('metrics/precision(B)', 0))
                training.recall = float(results.results_dict.get('metrics/recall(B)', 0))
                
        except Exception as e:
            training.status = 'failed'
            training.description = str(e)
            
    except Exception as e:
        if training:
            training.status = 'failed'
            training.description = str(e)
    finally:
        is_training = False
        stop_training = False
        if training:
            training.end_time = datetime.utcnow()
            db.session.commit()

# 添加YOLOv8训练回调类
class StopTrainingCallback:
    def __init__(self):
        self.model = None
        
    def before_train(self, trainer):
        self.model = trainer.model
        
    def after_epoch(self, trainer):
        global stop_training
        if stop_training:
            trainer.epoch_time.last = 0  # 强制结束训练
            trainer.epoch = trainer.epochs  # 设置当前轮数为最大轮数
            trainer.stop = True  # 停止训练标志

@app.route('/stop_training', methods=['POST'])
@require_login
def stop_training():
    global is_training, stop_training
    
    if not is_training:
        return jsonify({'status': 'error', 'message': '没有正在进行的训练任务'})
    
    # 设置停止标志
    stop_training = True
    
    # 更新最近的训练记录
    training = ModelTraining.query.filter_by(
        user_id=session['user_id'],
        status='training'
    ).order_by(ModelTraining.start_time.desc()).first()
    
    if training:
        training.status = 'stopping'
        training.description = '正在停止训练...'
        db.session.commit()
    
    return jsonify({'status': 'success', 'message': '正在停止训练，请稍候...'})

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        dataset_name = request.form.get('dataset_name')
        if not dataset_name:
            return jsonify({'status': 'error', 'message': '请提供数据集名称'})
        
        # 安全处理文件名
        dataset_name = secure_filename(dataset_name)
        
        # 创建数据集目录
        dataset_path = os.path.join('datasets', dataset_name)
        if os.path.exists(dataset_path):
            return jsonify({'status': 'error', 'message': '数据集名称已存在'})
        
        # 创建目录结构
        for dir_type in ['images', 'labels']:
            for split in ['train', 'val']:
                os.makedirs(os.path.join(dataset_path, dir_type, split), exist_ok=True)
        
        # 处理上传的文件
        file_types = {
            'images_train': ('images', 'train'),
            'labels_train': ('labels', 'train'),
            'images_val': ('images', 'val'),
            'labels_val': ('labels', 'val')
        }
        
        uploaded_files = {
            'images_train': [],
            'labels_train': [],
            'images_val': [],
            'labels_val': []
        }
        
        # 处理上传的文件
        for field in request.files:
            file = request.files[field]
            if not file.filename:
                continue
                
            # 从相对路径中提取文件类型和数据集类型
            relative_path = file.filename
            is_image = file.content_type.startswith('image/')
            is_label = relative_path.endswith('.txt')
            is_train = 'train' in relative_path.lower()
            
            if is_image:
                file_category = 'images_train' if is_train else 'images_val'
            elif is_label:
                file_category = 'labels_train' if is_train else 'labels_val'
            else:
                continue
            
            # 保存文件
            dir_type, split = file_types[file_category]
            filename = os.path.basename(relative_path)
            filename = secure_filename(filename)
            target_path = os.path.join(dataset_path, dir_type, split, filename)
            file.save(target_path)
            uploaded_files[file_category].append(filename)
        
        # 验证数据集结构
        train_images = len(uploaded_files['images_train'])
        train_labels = len(uploaded_files['labels_train'])
        val_images = len(uploaded_files['images_val'])
        val_labels = len(uploaded_files['labels_val'])
        
        if not all([train_images, train_labels, val_images, val_labels]):
            shutil.rmtree(dataset_path)
            return jsonify({
                'status': 'error',
                'message': '数据集上传不完整，请确保训练集和验证集都包含图片和标签文件'
            })
        
        if train_images != train_labels or val_images != val_labels:
            shutil.rmtree(dataset_path)
            return jsonify({
                'status': 'error',
                'message': f'图片和标签数量不匹配\n训练集：{train_images}张图片，{train_labels}个标签\n验证集：{val_images}张图片，{val_labels}个标签'
            })
        
        return jsonify({
            'status': 'success',
            'message': '数据集上传成功',
            'data': {
                'name': dataset_name,
                'train_count': train_images,
                'val_count': val_images
            }
        })
        
    except Exception as e:
        # 如果出错，清理已创建的目录
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        return jsonify({
            'status': 'error',
            'message': f'上传失败：{str(e)}'
        })

@app.route('/train', methods=['GET', 'POST'])
@require_login
def train():
    global training_thread
    
    if request.method == 'POST':
        if is_training:
            return jsonify({'status': 'error', 'message': '已有训练任务在进行中'})
        
        model_type = request.form.get('model_type', 'n')
        epochs = int(request.form.get('epochs', 100))
        batch_size = int(request.form.get('batch_size', 16))
        learning_rate = float(request.form.get('learning_rate', 0.01))
        dataset_path = request.form.get('dataset_path')
        
        if not dataset_path:
            return jsonify({'status': 'error', 'message': '请选择数据集'})
        
        # 验证数据集路径
        if not os.path.exists(dataset_path):
            return jsonify({'status': 'error', 'message': '数据集路径不存在'})
        
        # 在新线程中开始训练
        training_thread = threading.Thread(
            target=train_model,
            args=(model_type, epochs, batch_size, learning_rate, session['user_id'], dataset_path)
        )
        training_thread.start()
        
        return jsonify({'status': 'success', 'message': '训练已开始'})
    
    # 获取可用的数据集列表
    datasets = []
    # 定义可能的数据集目录
    dataset_roots = [
        'datasets',  # 默认数据集目录
        'cowFaceDetection-main/dataset',  # 项目数据集目录
        'data',  # 其他可能的数据集目录
    ]
    
    # 遍历所有可能的数据集目录
    for root in dataset_roots:
        if os.path.exists(root):
            # 遍历目录下的所有文件夹
            for item in os.listdir(root):
                item_path = os.path.join(root, item)
                if os.path.isdir(item_path):
                    # 检查是否是有效的数据集（包含images和labels子目录）
                    images_dir = os.path.join(item_path, 'images')
                    labels_dir = os.path.join(item_path, 'labels')
                    train_dir = os.path.join(images_dir, 'train')
                    val_dir = os.path.join(images_dir, 'val')
                    
                    is_valid = all(os.path.exists(d) for d in [images_dir, labels_dir, train_dir, val_dir])
                    
                    if is_valid:
                        # 计算数据集大小
                        train_images = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        val_images = len([f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        
                        datasets.append({
                            'name': item,
                            'path': item_path,
                            'train_count': train_images,
                            'val_count': val_images,
                            'root': root
                        })
    
    return render_template('train.html', datasets=datasets)

@app.route('/training_status')
@require_login
def training_status():
    # 获取最近的训练记录
    trainings = ModelTraining.query.filter_by(user_id=session['user_id'])\
        .order_by(ModelTraining.start_time.desc()).limit(5).all()
    
    current_status = {
        'is_training': is_training,
        'can_cancel': is_training and training_thread and training_thread.is_alive()
    }
    
    return render_template('training_status.html', trainings=trainings, current_status=current_status)

@app.route('/cancel_training', methods=['POST'])
@require_login
def cancel_training():
    global is_training
    if not is_training:
        return jsonify({'status': 'error', 'message': '没有正在进行的训练任务'})
    
    # 设置标志位来停止训练
    is_training = False
    
    # 更新最近的训练记录
    training = ModelTraining.query.filter_by(
        user_id=session['user_id'],
        status='training'
    ).order_by(ModelTraining.start_time.desc()).first()
    
    if training:
        training.status = 'cancelled'
        training.end_time = datetime.utcnow()
        training.description = '用户取消训练'
        db.session.commit()
    
    return jsonify({'status': 'success', 'message': '训练已取消'})

# change
def create_dataset_yaml(dataset_path):
    """创建数据集的yaml配置文件"""
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    
    # 检查数据集目录结构
    train_path = os.path.join(dataset_path, 'images', 'train')
    val_path = os.path.join(dataset_path, 'images', 'val')
    
    if not all(os.path.exists(p) for p in [train_path, val_path]):
        raise ValueError('数据集目录结构不正确，需要包含 images/train 和 images/val 目录')
    
    # 检查标签目录
    train_labels = os.path.join(dataset_path, 'labels', 'train')
    val_labels = os.path.join(dataset_path, 'labels', 'val')
    
    if not all(os.path.exists(p) for p in [train_labels, val_labels]):
        raise ValueError('数据集目录结构不正确，需要包含 labels/train 和 labels/val 目录')
    
    # 创建yaml配置
    config = {
        'path': dataset_path,
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'names': ['cow_face']  # 类别名称
    }
    
    # 保存配置文件
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    return yaml_path
