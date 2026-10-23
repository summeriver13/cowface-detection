from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from ..extensions import db
from ..models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username")).first()
        if user and check_password_hash(user.password, request.form.get("password") or ""):
            session["user_id"] = user.user_id
            flash("登录成功", "success")
            return redirect(url_for("web.dashboard.dashboard"))

        flash("用户名或密码错误", "danger")
    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("用户名和密码都不能为空", "danger")
            return redirect(url_for("web.auth.register"))
        if User.query.filter_by(username=username).first():
            flash("用户名已存在", "danger")
            return redirect(url_for("web.auth.register"))

        db.session.add(User(username=username, password=generate_password_hash(password)))
        db.session.commit()
        flash("注册成功，请登录", "success")
        return redirect(url_for("web.auth.login"))

    return render_template("register.html")


@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("已退出登录", "info")
    return redirect(url_for("web.auth.login"))
