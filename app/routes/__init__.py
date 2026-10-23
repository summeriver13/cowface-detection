from .auth import auth_bp
from .cattle import cattle_bp
from .detect import detect_bp
from .index import index_bp
from .train import train_bp

__all__ = ["auth_bp", "cattle_bp", "detect_bp", "index_bp", "train_bp"]
