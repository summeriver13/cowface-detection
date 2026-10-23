from ..extensions import db


# 牛只个体信息表
class Cattle(db.Model):
    __tablename__ = 'cattle'
    cattle_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    records = db.relationship('RecognitionRecord', backref='cattle', lazy=True)
