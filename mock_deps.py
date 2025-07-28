# Minimal mocks for dependencies to enable testing

class SettingsConfigDict:
    def __init__(self, **kwargs):
        pass

class BaseSettings:
    def __init__(self, **kwargs):
        # Set default values for required environment variables
        self.FLASK_PORT = 9000
        self.ALPACA_API_KEY = "test_key"
        self.ALPACA_SECRET_KEY = "test_secret" 
        self.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
        self.ALPACA_DATA_FEED = "iex"
        self.FINNHUB_API_KEY = None
        self.FUNDAMENTAL_API_KEY = None
        self.NEWS_API_KEY = None
        self.IEX_API_TOKEN = None
        self.BOT_MODE = "balanced"
        self.MODEL_PATH = "trained_model.pkl"
        self.HALT_FLAG_PATH = "halt.flag"
        self.MAX_PORTFOLIO_POSITIONS = 20
        self.LIMIT_ORDER_SLIPPAGE = 0.005
        self.HEALTHCHECK_PORT = 8081
        self.RUN_HEALTHCHECK = "0"
        self.BUY_THRESHOLD = 0.5
        self.WEBHOOK_SECRET = "test_webhook_secret"
        self.WEBHOOK_PORT = 9000
        self.SLIPPAGE_THRESHOLD = 0.003
        self.REBALANCE_INTERVAL_MIN = 1440
        self.SHADOW_MODE = False
        self.DRY_RUN = False
        self.DISABLE_DAILY_RETRAIN = False
        self.TRADE_LOG_FILE = "data/trades.csv"
        self.FORCE_TRADES = False
        self.DISASTER_DD_LIMIT = 0.2
        self.MODEL_RF_PATH = "model_rf.pkl"
        self.MODEL_XGB_PATH = "model_xgb.pkl"
        self.MODEL_LGB_PATH = "model_lgb.pkl"
        self.RL_MODEL_PATH = "rl_agent.zip"
        self.USE_RL_AGENT = False
        self.SECTOR_EXPOSURE_CAP = 0.4
        self.MAX_OPEN_POSITIONS = 10
        self.WEEKLY_DRAWDOWN_LIMIT = 0.15
        self.VOLUME_THRESHOLD = 50000
        self.DOLLAR_RISK_LIMIT = 0.02
        self.FINNHUB_RPM = 60
        self.MINUTE_CACHE_TTL = 60
        self.EQUITY_EXPOSURE_CAP = 2.5
        self.PORTFOLIO_EXPOSURE_CAP = 2.5
        self.SEED = 42
        self.RATE_LIMIT_BUDGET = 190
        
        # Apply any provided overrides
        for key, value in kwargs.items():
            setattr(self, key, value)

def __getattr__(name):
    if name == 'BaseSettings':
        return BaseSettings
    elif name == 'SettingsConfigDict':
        return SettingsConfigDict
    raise AttributeError(f"module 'mock_deps' has no attribute '{name}'")
