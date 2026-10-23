from pathlib import Path

from flask import Flask

from . import config
from .extensions import db
from .routes import auth_bp, cattle_bp, detect_bp, index_bp, train_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    with app.app_context():
        from . import models  # noqa: F401  建表前先注册 ORM 元数据
        db.create_all()

    for blueprint in (auth_bp, index_bp, cattle_bp, detect_bp, train_bp):
        app.register_blueprint(blueprint)

    return app
