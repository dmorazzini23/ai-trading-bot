"""CLI entrypoint so ``python -m retrain`` resolves correctly."""

from ai_trading.retrain import main

if __name__ == "__main__":  # pragma: no cover - CLI execution helper
    raise SystemExit(main())
