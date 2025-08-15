import importlib.util
import pathlib

if __spec__ is None:  # pragma: no cover
    __spec__ = importlib.util.spec_from_file_location("validate_env", pathlib.Path(__file__).resolve())


def _main() -> None:
    from ai_trading.config.management import TradingConfig
    TradingConfig.from_env().validate_environment()
    print("Environment OK")


if __name__ == "__main__":  # pragma: no cover
    _main()
