from flask import Blueprint, Flask, jsonify, request
from werkzeug.exceptions import HTTPException, InternalServerError

API_PREFIX = "/api/"
api_bp = Blueprint("api", __name__, url_prefix="/api")


def ok(data=None, status: int = 200):
    body: dict = {"ok": True}
    if data is not None:
        body["data"] = data
    return jsonify(body), status


def fail(code: str, message: str, status: int = 400):
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def _is_api_request() -> bool:
    return request.path.startswith(API_PREFIX)


def install_error_handlers(app: Flask) -> None:
    """让 /api 下的错误统一走 {"ok": false} 信封，页面请求保持默认行为。"""

    @app.errorhandler(HTTPException)
    def render_http_error(exc: HTTPException):
        if not _is_api_request():
            return exc
        code = (exc.name or "error").lower().replace(" ", "_")
        return fail(code, exc.description or "", exc.code or 500)

    @app.errorhandler(Exception)
    def render_unexpected_error(exc: Exception):
        if isinstance(exc, HTTPException):
            return render_http_error(exc)
        app.logger.exception("unhandled error on %s", request.path)
        if _is_api_request():
            return fail("internal_error", "服务端错误", 500)
        return InternalServerError()


from . import cattle, detect, train  # noqa: E402  子蓝图要在 api_bp 定义之后导入

api_bp.register_blueprint(cattle.cattle_api)
api_bp.register_blueprint(detect.detect_api)
api_bp.register_blueprint(train.train_api)

__all__ = ["api_bp", "fail", "install_error_handlers", "ok"]
