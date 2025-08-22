# tools/ci/tighten_settings_types.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "ai_trading" / "config" / "settings.py"

def tighten():
    txt = SETTINGS.read_text(encoding="utf-8", errors="ignore")

    # Replace specific str | None fields with proper types and defaults
    replacements = [
        # Booleans
        (r'    force_trades: str \| None = Field\(default=None, env=\'FORCE_TRADES\'\)',
         '    force_trades: bool = Field(False, env=\'FORCE_TRADES\')'),
        (r'    use_rl_agent: str \| None = Field\(default=None, env=\'USE_RL_AGENT\'\)',
         '    use_rl_agent: bool = Field(False, env=\'USE_RL_AGENT\')'),
        (r'    verbose: str \| None = Field\(default=None, env=\'VERBOSE\'\)',
         '    verbose: bool = Field(False, env=\'VERBOSE\')'),
        (r'    verbose_logging: str \| None = Field\(default=None, env=\'VERBOSE_LOGGING\'\)',
         '    verbose_logging: bool = Field(False, env=\'VERBOSE_LOGGING\')'),

        # Integers
        (r'    pyramid_levels: str \| None = Field\(default=None, env=\'PYRAMID_LEVELS\'\)',
         '    pyramid_levels: int = Field(0, env=\'PYRAMID_LEVELS\')'),
        (r'    min_health_rows: str \| None = Field\(default=None, env=\'MIN_HEALTH_ROWS\'\)',
         '    min_health_rows: int = Field(100, env=\'MIN_HEALTH_ROWS\')'),

        # Floats
        (r'    ml_confidence_threshold: str \| None = Field\(default=None, env=\'ML_CONFIDENCE_THRESHOLD\'\)',
         '    ml_confidence_threshold: float = Field(0.6, env=\'ML_CONFIDENCE_THRESHOLD\')'),
        (r'    portfolio_drift_threshold: str \| None = Field\(default=None, env=\'PORTFOLIO_DRIFT_THRESHOLD\'\)',
         '    portfolio_drift_threshold: float = Field(0.05, env=\'PORTFOLIO_DRIFT_THRESHOLD\')'),
        (r'    max_drawdown_threshold: str \| None = Field\(default=None, env=\'MAX_DRAWDOWN_THRESHOLD\'\)',
         '    max_drawdown_threshold: float = Field(0.15, env=\'MAX_DRAWDOWN_THRESHOLD\')'),
        (r'    volume_spike_threshold: str \| None = Field\(default=None, env=\'VOLUME_SPIKE_THRESHOLD\'\)',
         '    volume_spike_threshold: float = Field(1.5, env=\'VOLUME_SPIKE_THRESHOLD\')'),

        # Check for scheduler_sleep_seconds and update if needed (currently int)
        (r'    scheduler_sleep_seconds: int = Field\(30, env="SCHEDULER_SLEEP_SECONDS"\)',
         '    scheduler_sleep_seconds: int = Field(5, env="SCHEDULER_SLEEP_SECONDS")'),

        # Update model_path to None default and trade_log_file default
        (r'    model_path: str = Field\("meta_model\.pkl", env="MODEL_PATH"\)',
         '    model_path: str | None = Field(None, env="MODEL_PATH")'),
        (r'    trade_log_file: str = Field\("test_trades\.csv", env="TRADE_LOG_FILE"\)',
         '    trade_log_file: str = Field("trades.csv", env="TRADE_LOG_FILE")'),
    ]

    for pattern, replacement in replacements:
        txt = re.sub(pattern, replacement, txt)

    SETTINGS.write_text(txt, encoding="utf-8")

if __name__ == "__main__":
    tighten()
    print("Settings types tightened.")
