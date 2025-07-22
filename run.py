# Entry-point alias so tests can import "run"
# AI-AGENT-REF: delegate entrypoint to canonical main module
from ai_trading.main import main

if __name__ == "__main__":
    main()

