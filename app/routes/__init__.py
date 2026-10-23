from flask import Blueprint

web_bp = Blueprint("web", __name__)

from . import auth, cattle, dashboard, detect, train  # noqa: E402  子蓝图要在 web_bp 定义之后导入

for child in (auth.auth_bp, dashboard.dashboard_bp, cattle.cattle_bp, detect.detect_bp, train.train_bp):
    web_bp.register_blueprint(child)

__all__ = ["web_bp"]
