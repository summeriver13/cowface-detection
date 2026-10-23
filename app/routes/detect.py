from flask import Blueprint, render_template

from ..controllers import require_login

detect_bp = Blueprint("detect", __name__)


@detect_bp.route("/detect")
@require_login
def detect_page():
    return render_template("detect.html")
