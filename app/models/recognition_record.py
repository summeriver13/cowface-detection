from datetime import UTC, datetime

from ..extensions import db


def utcnow() -> datetime:
    """列类型是 naive DATETIME，统一存 UTC，展示时由前端换成本地时间。"""
    return datetime.now(UTC).replace(tzinfo=None)


class RecognitionRecord(db.Model):
    __tablename__ = 'recognition_record'
    record_id = db.Column(db.Integer, primary_key=True)
    cattle_id = db.Column(db.Integer, db.ForeignKey('cattle.cattle_id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=utcnow)
    # 一行里塞了检测置信度、识别置信度和类别，50 字符不够，会被截断
    result = db.Column(db.Text)
