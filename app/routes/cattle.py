from flask import Blueprint, flash, redirect, render_template, session, url_for
from sqlalchemy import func

from ..controllers import require_login
from ..extensions import db
from ..models import Cattle, RecognitionRecord

cattle_bp = Blueprint("cattle", __name__)


def _record_counts(user_id: int) -> dict[int, int]:
    rows = (
        db.session.query(RecognitionRecord.cattle_id, func.count(RecognitionRecord.record_id))
        .join(Cattle, Cattle.cattle_id == RecognitionRecord.cattle_id)
        .filter(Cattle.user_id == user_id)
        .group_by(RecognitionRecord.cattle_id)
        .all()
    )
    return {cattle_id: count for cattle_id, count in rows}


@cattle_bp.route("/cattle")
@require_login
def cattle_list():
    cattle = Cattle.query.filter_by(user_id=session["user_id"]).order_by(Cattle.cattle_id).all()
    counts = _record_counts(session["user_id"])
    for cow in cattle:
        cow.record_count = counts.get(cow.cattle_id, 0)
    return render_template("cattle.html", cattle=cattle)


@cattle_bp.route("/cattle/<int:cattle_id>")
@require_login
def cattle_detail(cattle_id: int):
    cattle = Cattle.query.filter_by(cattle_id=cattle_id, user_id=session["user_id"]).first()
    if cattle is None:
        flash("牛只不存在", "warning")
        return redirect(url_for("web.cattle.cattle_list"))

    records = (
        RecognitionRecord.query.filter_by(cattle_id=cattle_id)
        .order_by(RecognitionRecord.timestamp.desc())
        .all()
    )
    return render_template("cattle_detail.html", cattle=cattle, records=records)
