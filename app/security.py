import secrets

from flask import Flask, jsonify, request, session

SESSION_KEY = "csrf_token"
HEADER = "X-CSRF-Token"
FORM_FIELD = "csrf_token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
TOKEN_BYTES = 32


def csrf_token() -> str:
    """session 里的 token，模板和 API 头共用同一份。"""
    token = session.get(SESSION_KEY)
    if token is None:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        session[SESSION_KEY] = token
    return token


def _is_valid(sent: str | None) -> bool:
    expected = session.get(SESSION_KEY)
    if not expected or not sent:
        return False
    return secrets.compare_digest(sent, expected)


def init_csrf(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def reject_unsafe_request_without_token():
        if request.method in SAFE_METHODS:
            return None
        sent = request.headers.get(HEADER) or request.form.get(FORM_FIELD)
        if _is_valid(sent):
            return None
        return (
            jsonify({"ok": False, "error": {"code": "invalid_csrf", "message": "CSRF 校验失败，请刷新页面重试"}}),
            400,
        )
