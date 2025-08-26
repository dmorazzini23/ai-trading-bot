import subprocess, sys, os


def test_cli_dry_run_exits_zero_and_marks_indicator():
    env = dict(os.environ)
    # Ensure clean behavior regardless of absent .env
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Do not require network or heavy stacks in dry-run
    r = subprocess.run([sys.executable, "-m", "ai_trading", "--dry-run"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"non-zero exit: {r.returncode}\nSTDERR:\n{r.stderr}"
    combined = (r.stdout or "") + (r.stderr or "")
    assert "INDICATOR_IMPORT_OK" in combined, "missing INDICATOR_IMPORT_OK banner"
