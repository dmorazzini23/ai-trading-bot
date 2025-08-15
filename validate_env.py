from ai_trading.config import _require_env_vars, reload_env

# AI-AGENT-REF: allow runpy.run_module execution
if __spec__ is None:  # pragma: no cover
    import importlib.util, pathlib

    _p = pathlib.Path(__file__).resolve()
    __spec__ = importlib.util.spec_from_file_location("validate_env", _p)


def _main() -> int:
    reload_env()
    _require_env_vars("WEBHOOK_SECRET", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
