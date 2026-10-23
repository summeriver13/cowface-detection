from flask import Blueprint, request, session
from sqlalchemy.exc import SQLAlchemyError

from ..controllers import require_login
from ..extensions import db
from ..models import Cattle, RecognitionRecord
from . import fail, ok

cattle_api = Blueprint("cattle", __name__)

MAX_NAME_LENGTH = 64


def _owned_cattle(cattle_id: int) -> Cattle | None:
    return Cattle.query.filter_by(cattle_id=cattle_id, user_id=session["user_id"]).first()


def _read_name() -> str:
    payload = request.get_json(silent=True) or request.form
    return (payload.get("name") or "").strip()


def _validate_name(name: str):
    if not name:
        return fail("invalid_name", "牛只名称不能为空")
    if len(name) > MAX_NAME_LENGTH:
        return fail("invalid_name", f"牛只名称不能超过 {MAX_NAME_LENGTH} 个字符")
    return None


@cattle_api.post("/cattle")
@require_login
def create_cattle():
    name = _read_name()
    invalid = _validate_name(name)
    if invalid:
        return invalid

    cattle = Cattle(name=name, user_id=session["user_id"])
    db.session.add(cattle)
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        return fail("db_error", "添加失败", 500)
    return ok({"cattle_id": cattle.cattle_id}, 201)


@cattle_api.patch("/cattle/<int:cattle_id>")
@require_login
def rename_cattle(cattle_id: int):
    cattle = _owned_cattle(cattle_id)
    if cattle is None:
        return fail("not_found", "牛只不存在", 404)

    name = _read_name()
    invalid = _validate_name(name)
    if invalid:
        return invalid

    cattle.name = name
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        return fail("db_error", "修改失败", 500)
    return ok({"cattle_id": cattle.cattle_id, "name": cattle.name})


@cattle_api.delete("/cattle/<int:cattle_id>")
@require_login
def delete_cattle(cattle_id: int):
    cattle = _owned_cattle(cattle_id)
    if cattle is None:
        return fail("not_found", "牛只不存在", 404)

    try:
        RecognitionRecord.query.filter_by(cattle_id=cattle_id).delete()
        db.session.delete(cattle)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        return fail("db_error", "删除失败", 500)
    return ok({"cattle_id": cattle_id})
