from flask import Blueprint, flash, render_template, request, redirect, session, url_for
from ..controllers import require_login
from ..models import Cattle, RecognitionRecord
from ..extensions import db

cattle_bp = Blueprint('cattle', __name__)

@cattle_bp.route('/cattle', methods=['GET', 'POST'])
@require_login
def cattle():
    if request.method == 'POST':
        name = request.form.get('name')
        cattle = Cattle(
            name=name,
            user_id=session['user_id']
        )
        db.session.add(cattle)
        db.session.commit()
        flash('添加成功')
        return redirect(url_for('cattle.cattle'))
    
    # 获取当前用户的所有牛只及其记录
    cattle_list = Cattle.query.filter_by(
        user_id=session['user_id']
    ).all()
    
    # 预加载每个牛只的最近5条记录
    for cow in cattle_list:
        cow.recent_records = RecognitionRecord.query.filter_by(
            cattle_id=cow.cattle_id
        ).order_by(
            RecognitionRecord.timestamp.desc()
        ).limit(5).all()
    
    return render_template('cattle.html', cattle=cattle_list)
