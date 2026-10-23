from pathlib import Path

from flask import Flask

from . import config
from .api import api_bp, install_error_handlers
from .extensions import db
from .routes import web_bp
from .security import init_csrf


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    init_csrf(app)
    with app.app_context():
        from . import models  # noqa: F401  建表前先注册 ORM 元数据
        db.create_all()

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    install_error_handlers(app)

    return app
