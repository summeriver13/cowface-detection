from .auth import auth_bp
from .index import index_bp
from .cattle import cattle_bp
from .detect import detect_bp
from .train import train_bp

def register_blueprints(app):
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(index_bp)
    app.register_blueprint(cattle_bp, url_prefix="/cattle")
    app.register_blueprint(detect_bp, url_prefix="/detect")
    app.register_blueprint(train_bp)
