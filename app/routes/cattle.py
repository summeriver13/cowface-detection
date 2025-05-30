from collections import defaultdict
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
    cattle_list = Cattle.query.filter_by(user_id=session['user_id']).all()
    cattle_ids = [c.cattle_id for c in cattle_list]
    
    # 一次查询获取所有牛只最近5条记录
    records = RecognitionRecord.query.filter(
        RecognitionRecord.cattle_id.in_(cattle_ids)
    ).order_by(RecognitionRecord.cattle_id, RecognitionRecord.timestamp.desc()).all()
    
    # 按cattle_id分组，取每组前5条
    grouped = defaultdict(list)
    for r in records:
        if len(grouped[r.cattle_id]) < 5:
            grouped[r.cattle_id].append(r)


    # 给每个牛只赋值最近5条记录
    for cow in cattle_list:
        cow.recent_records = grouped.get(cow.cattle_id, [])

    
    return render_template('cattle.html', cattle=cattle_list)

@cattle_bp.route('/cattle/edit/<int:cattle_id>', methods=['GET', 'POST'])
@require_login
def edit_cattle(cattle_id):
    cattle = Cattle.query.filter_by(cattle_id=cattle_id, user_id=session['user_id']).first_or_404()
    if request.method == 'POST':
        name = request.form.get('name')
        if name:
            cattle.name = name
            db.session.commit()
            flash('更新成功')
            return redirect(url_for('cattle.cattle'))
    return render_template('edit_cattle.html', cattle=cattle)

@cattle_bp.route('/cattle/delete/<int:cattle_id>', methods=['POST'])
@require_login
def delete_cattle(cattle_id):
    cattle = Cattle.query.filter_by(cattle_id=cattle_id, user_id=session['user_id']).first_or_404()
    # 先删除该牛的所有识别记录
    RecognitionRecord.query.filter_by(cattle_id=cattle.cattle_id).delete()
    # 再删除牛只
    db.session.delete(cattle)
    db.session.commit()
    flash('删除成功')
    return redirect(url_for('cattle.cattle'))