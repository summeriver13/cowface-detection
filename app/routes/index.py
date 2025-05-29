from flask import Blueprint, session, render_template, redirect, url_for
from ..models import User
from ..controllers import require_login  # 装饰器放auth.py

index_bp = Blueprint('index', __name__)

# 修改根路由
@index_bp.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('index.dashboard'))
    return redirect(url_for('auth.login'))

# 修改仪表盘路由，添加登录验证
@index_bp.route('/dashboard')
@require_login
def dashboard():
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)
