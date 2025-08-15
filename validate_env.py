from ai_trading.config import _require_env_vars, reload_env


def _main() -> int:
    reload_env()
    _require_env_vars("WEBHOOK_SECRET", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
