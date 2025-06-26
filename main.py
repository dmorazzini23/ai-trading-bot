"""Alias to run module for backward compatibility."""
import importlib

import run as _run

_run = importlib.reload(_run)

create_flask_app = _run.create_flask_app
run_flask_app = _run.run_flask_app
run_bot = _run.run_bot
validate_environment = _run.validate_environment
main = _run.main

if __name__ == "__main__":
    _run.main()
