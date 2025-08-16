import os

# AI-AGENT-REF: minimal env validator for runpy execution
REQUIRED_KEYS = (
    "WEBHOOK_SECRET",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
)


def _main() -> bool:
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    return True


if __name__ == "__main__":  # pragma: no cover - manual execution
    _main()
