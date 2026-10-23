from flask import Blueprint, render_template, session

from ..config import MODEL_TYPE_LABELS
from ..controllers import require_login
from ..models import ModelTraining

train_bp = Blueprint("train", __name__)


@train_bp.route("/train")
@require_login
def train_page():
    return render_template("train.html", model_types=MODEL_TYPE_LABELS.items())


@train_bp.route("/train/status")
@require_login
def training_status_page():
    trainings = (
        ModelTraining.query.filter_by(user_id=session["user_id"])
        .order_by(ModelTraining.start_time.desc())
        .all()
    )
    return render_template("training_status.html", trainings=trainings, model_types=MODEL_TYPE_LABELS)
