from functools import wraps
from flask import session, flash, redirect, url_for

# 登录验证装饰器
def require_login(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function
