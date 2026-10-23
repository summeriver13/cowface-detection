from flask import Blueprint, redirect, render_template, session, url_for
from sqlalchemy import func

from ..controllers import require_login
from ..extensions import db
from ..models import Cattle, ModelTraining, RecognitionRecord, User

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("web.dashboard.dashboard"))
    return redirect(url_for("web.auth.login"))


@dashboard_bp.route("/dashboard")
@require_login
def dashboard():
    user_id = session["user_id"]
    stats = {
        "cattle": db.session.query(func.count(Cattle.cattle_id)).filter(Cattle.user_id == user_id).scalar(),
        "records": db.session.query(func.count(RecognitionRecord.record_id))
        .join(Cattle, Cattle.cattle_id == RecognitionRecord.cattle_id)
        .filter(Cattle.user_id == user_id)
        .scalar(),
        "trainings": db.session.query(func.count(ModelTraining.training_id))
        .filter(ModelTraining.user_id == user_id)
        .scalar(),
    }
    user = db.session.get(User, user_id)
    return render_template("dashboard.html", user=user, stats=stats)
