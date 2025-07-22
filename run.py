# Entry-point alias so tests can import "run"
# AI-AGENT-REF: delegate entrypoint to package __main__
from ai_trading.__main__ import main

if __name__ == "__main__":
    main()

