from functools import wraps

from flask import flash, jsonify, redirect, request, session, url_for


def _expects_json() -> bool:
    blueprint = request.blueprint or ""
    return blueprint.split(".")[0] == "api" or request.path.startswith("/api/")


def require_login(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if session.get("user_id"):
            return view(*args, **kwargs)
        if _expects_json():
            return jsonify({"ok": False, "error": {"code": "unauthorized", "message": "请先登录"}}), 401
        flash("请先登录", "warning")
        return redirect(url_for("web.auth.login"))

    return wrapper
