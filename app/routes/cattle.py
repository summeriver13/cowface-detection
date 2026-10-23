from flask import Blueprint, flash, jsonify, render_template, request, redirect, session, url_for
from sqlalchemy.exc import SQLAlchemyError

from ..controllers import require_login
from ..models import Cattle, RecognitionRecord
from ..extensions import db

cattle_bp = Blueprint('cattle', __name__)

@cattle_bp.route('/cattle', methods=['GET'])
@require_login
def cattle():
    # 获取当前用户的所有牛只及其记录
    cattle_list = Cattle.query.filter_by(user_id=session['user_id']).all()

     # 预加载每个牛只的最近5条记录
    for cow in cattle_list:
        cow.recent_records = RecognitionRecord.query.filter_by(
            cattle_id=cow.cattle_id
        ).order_by(RecognitionRecord.timestamp.desc()).limit(5).all()
    
    return render_template('cattle.html', cattle=cattle_list)

@cattle_bp.route('/cattle/add', methods=['POST'])
@require_login
def add_cattle():
    name = request.form.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': '牛只名称不能为空'})
    
    cattle = Cattle(
        name=name,
        user_id=session['user_id']
    )
    db.session.add(cattle)
    try:
        db.session.commit()
        return jsonify({'status': 'success', 'message': '添加成功', 'cattle_id': cattle.cattle_id})
    except SQLAlchemyError:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '添加失败'})


@cattle_bp.route('/cattle/edit/<int:cattle_id>', methods=['POST'])
@require_login
def edit_cattle(cattle_id):
    cattle = Cattle.query.filter_by(cattle_id=cattle_id, user_id=session['user_id']).first()
    if not cattle:
        return jsonify({'status': 'error', 'message': '牛只不存在'})
    
    name = request.form.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': '牛只名称不能为空'})
    
    cattle.name = name
    try:
        db.session.commit()
        return jsonify({'status': 'success', 'message': '修改成功'})
    except SQLAlchemyError:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '修改失败'})



@cattle_bp.route('/cattle/delete/<int:cattle_id>', methods=['POST'])
@require_login
def delete_cattle(cattle_id):
    cattle = Cattle.query.filter_by(cattle_id=cattle_id, user_id=session['user_id']).first()
    if not cattle:
        return jsonify({'status': 'error', 'message': '牛只不存在'})
    
    try:
        # 首先删除关联的识别记录
        RecognitionRecord.query.filter_by(cattle_id=cattle_id).delete()
        # 然后删除牛只记录
        db.session.delete(cattle)
        db.session.commit()
        return jsonify({'status': 'success', 'message': '删除成功'})
    except SQLAlchemyError:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '删除失败'})

@cattle_bp.route('/cattle/detail/<int:cattle_id>')
@require_login
def cattle_detail(cattle_id):
    cattle = Cattle.query.filter_by(cattle_id=cattle_id, user_id=session['user_id']).first()
    if not cattle:
        flash('牛只不存在')
        return redirect(url_for('cattle.cattle'))
    
    # 获取该牛只的所有识别记录
    records = RecognitionRecord.query.filter_by(cattle_id=cattle_id)\
        .order_by(RecognitionRecord.timestamp.desc()).all()
    
    return render_template('cattle_detail.html', cattle=cattle, records=records)