import os

class Config:
    def __init__(self):
        self.verbose = os.environ.get("SGP_VERBOSE", "false").lower() in ["true", "1", "yes", "on"]
        self.numba_enabled = os.environ.get("NUMBA_DISABLE_JIT", "0") == "0"
        self.password = ""          # 密码验证跳过
        self.language = "en"       # 默认英语

config = Config()