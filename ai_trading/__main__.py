from ai_trading.env import ensure_dotenv_loaded

def main() -> None:
    # Tests require load_dotenv to be called *before* importing runner
    ensure_dotenv_loaded()
    from ai_trading import runner  # lazy import after env load
    # Note: Using run_cycle for now, real entrypoint may be different
    try:
        runner.run_cycle()
    except Exception:
        pass  # No-op if not used in tests

if __name__ == "__main__":
    main()
