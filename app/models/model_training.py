import datetime

from ..extensions import db


# 训练模型表
class ModelTraining(db.Model):
    __tablename__ = 'model_training'
    training_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    model_type = db.Column(db.String(10), nullable=False)  # n, s, m, l, x
    epochs = db.Column(db.Integer, nullable=False)
    batch_size = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    
    start_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(20))  # training, completed, failed
    map50 = db.Column(db.Float)
    map50_95 = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    description = db.Column(db.Text)

    # 添加关系
    user = db.relationship('User', backref=db.backref('trainings', lazy=True))