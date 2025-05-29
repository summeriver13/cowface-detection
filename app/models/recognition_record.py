from ..extensions import db
import datetime

class RecognitionRecord(db.Model):
    __tablename__ = 'recognition_record'
    record_id = db.Column(db.Integer, primary_key=True)
    cattle_id = db.Column(db.Integer, db.ForeignKey('cattle.cattle_id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.timezone.utc)
    result = db.Column(db.String(50))  # 识别结果