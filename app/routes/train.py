from flask import Blueprint, flash, render_template, request, redirect, session, url_for
from ..controllers import require_login
from ..models import ModelTraining
from ..extensions import db
import threading

train_bp = Blueprint('train-routes', __name__)

@train_bp.route('/train', methods=['GET', 'POST'])
@require_login
def train():
    if request.method == 'POST':
        try:
            model_type = request.form.get('model_type')
            epochs = int(request.form.get('epochs'))
            batch_size = int(request.form.get('batch_size'))
            learning_rate = float(request.form.get('learning_rate'))
            
            # 验证参数
            if not all([model_type, epochs, batch_size, learning_rate]):
                flash('请填写所有训练参数')
                return redirect(url_for('train'))
            
            # 确保目录存在
            from pathlib import Path
            Path('ml_models/pretrained_models').mkdir(parents=True, exist_ok=True)
            Path('runs/train').mkdir(parents=True, exist_ok=True)
            
            # 在后台启动训练
            from flask import current_app
            from ..controllers import train_model
            thread = threading.Thread(
                target=train_model,
                args=(current_app._get_current_object(), model_type, epochs, batch_size, learning_rate, session['user_id'])
            )
            thread.daemon = True  # 设置为守护线程
            thread.start()
            

            flash('模型训练已开始，请在训练状态页面查看进度')
            return redirect(url_for('train-routes.training_status'))
            
        except Exception as e:
            print(f"Training error: {str(e)}")  # 打印错误信息
            flash(f'启动训练失败: {str(e)}')
            return redirect(url_for('train-routes.train'))
    
    return render_template('train.html')

@train_bp.route('/training_status')
@require_login
def training_status():
    trainings = ModelTraining.query.filter_by(user_id=session['user_id']).order_by(ModelTraining.start_time.desc()).all()
    return render_template('training_status.html', trainings=trainings)

@train_bp.route('/delete_training/<int:training_id>', methods=['POST'])
@require_login
def delete_training(training_id):
    training = ModelTraining.query.get(training_id)
    if not training or training.user_id != session['user_id']:
        flash('训练记录不存在或无权限删除')
        return redirect(url_for('train-routes.training_status'))
    
    try:
        db.session.delete(training)
        db.session.commit()
        flash('训练记录已删除')
    except Exception as e:
        db.session.rollback()
        flash(f'删除失败: {str(e)}')
    
    return redirect(url_for('train-routes.training_status'))