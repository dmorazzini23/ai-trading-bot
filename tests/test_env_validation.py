import subprocess, sys, os


def test_empty_interval_is_handled_gracefully():
    # Simulate the bad case: env var defined but empty
    env = dict(os.environ)
    env["AI_TRADING_INTERVAL"] = ""
    env.setdefault("PYTHONFAULTHANDLER", "1")
    r = subprocess.run([sys.executable, "-m", "ai_trading", "--dry-run"],
                       env=env, capture_output=True, text=True)
    # We allow either a clean exit or a handled error, but not an unhandled traceback
    combined = (r.stdout or "") + (r.stderr or "")
    assert "Traceback" not in combined, f"Unhandled exception:\n{combined}"
