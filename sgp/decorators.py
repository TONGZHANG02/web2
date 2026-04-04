import functools
import time
from sgp.config import config

def timing(func=None, *, target=print):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if config.verbose:
                start = time.time()
                result = func(*args, **kw)
                duration = time.time() - start
                target(f"Function {func.__qualname__: >25s} took {duration:2.4f} seconds")
                return result
            else:
                return func(*args, **kw)
        return wrapper
    return decorator if func is None else decorator(func)