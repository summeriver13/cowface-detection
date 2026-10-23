import threading

from flask import Blueprint, current_app, request, session

from ..config import MODEL_TYPE_LABELS
from ..controllers import require_login, train_model
from ..extensions import db
from ..models import ModelTraining
from . import fail, ok

train_api = Blueprint("train", __name__)

EPOCH_RANGE = (1, 1000)
BATCH_RANGE = (1, 128)
LR_RANGE = (1e-5, 1.0)


def _read_int(field: str) -> int | None:
    try:
        return int(request.form[field])
    except (KeyError, ValueError):
        return None


def _parse_request():
    model_type = request.form.get("model_type")
    epochs = _read_int("epochs")
    batch_size = _read_int("batch_size")
    try:
        learning_rate = float(request.form["learning_rate"])
    except (KeyError, ValueError):
        learning_rate = None

    if model_type not in MODEL_TYPE_LABELS:
        return None, fail("invalid_model_type", f"模型类型必须是 {'/'.join(MODEL_TYPE_LABELS)}")
    if epochs is None or not EPOCH_RANGE[0] <= epochs <= EPOCH_RANGE[1]:
        return None, fail("invalid_epochs", f"epochs 必须在 {EPOCH_RANGE[0]}~{EPOCH_RANGE[1]} 之间")
    if batch_size is None or not BATCH_RANGE[0] <= batch_size <= BATCH_RANGE[1]:
        return None, fail("invalid_batch_size", f"batch size 必须在 {BATCH_RANGE[0]}~{BATCH_RANGE[1]} 之间")
    if learning_rate is None or not LR_RANGE[0] <= learning_rate <= LR_RANGE[1]:
        return None, fail("invalid_learning_rate", "学习率超出允许范围")
    return (model_type, epochs, batch_size, learning_rate), None


@train_api.get("/train")
@require_login
def list_trainings():
    trainings = (
        ModelTraining.query.filter_by(user_id=session["user_id"])
        .order_by(ModelTraining.start_time.desc())
        .limit(50)
        .all()
    )
    return ok(
        {
            "trainings": [
                {
                    "training_id": item.training_id,
                    "model_type": item.model_type,
                    "status": item.status,
                    "start_time": item.start_time.isoformat() if item.start_time else None,
                    "end_time": item.end_time.isoformat() if item.end_time else None,
                    "map50": item.map50,
                    "map50_95": item.map50_95,
                    "precision": item.precision,
                    "recall": item.recall,
                    "description": item.description,
                }
                for item in trainings
            ]
        }
    )


@train_api.post("/train")
@require_login
def start_training():
    params, error = _parse_request()
    if error:
        return error

    model_type, epochs, batch_size, learning_rate = params
    app = current_app._get_current_object()
    user_id = session["user_id"]
    thread = threading.Thread(
        target=train_model,
        args=(app, model_type, epochs, batch_size, learning_rate, user_id),
        daemon=True,
    )
    thread.start()
    return ok({"status": "training"}, 202)


@train_api.delete("/train/<int:training_id>")
@require_login
def delete_training(training_id: int):
    training = db.session.get(ModelTraining, training_id)
    if training is None or training.user_id != session["user_id"]:
        return fail("not_found", "训练记录不存在", 404)

    try:
        db.session.delete(training)
        db.session.commit()
    except Exception:
        db.session.rollback()
        return fail("db_error", "删除失败", 500)
    return ok({"training_id": training_id})
