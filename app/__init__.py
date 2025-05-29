from flask import Flask

from .extensions import db
from .config import *
from .routes import auth_bp, index_bp, cattle_bp, detect_bp, train_bp

def create_app():
    app = Flask(__name__)
    # app configration
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
    
    # binding database
    db.init_app(app)
    
    # 注册蓝图
    app.register_blueprint(auth_bp)
    app.register_blueprint(index_bp)
    app.register_blueprint(cattle_bp)
    app.register_blueprint(detect_bp)
    app.register_blueprint(train_bp)

    return app