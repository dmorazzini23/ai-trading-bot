#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime, timedelta

LOG_FILE = "/var/log/ai-trading-scheduler.log"
REPORT = "health_report.txt"


def main():
    # 1) Check for uncaught exceptions in last 24h
    since = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    grep = subprocess.run(
        ["grep", "-R", "Traceback", "--include", "*.log", LOG_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    errors = grep.stdout.decode().strip()

    # 2) Run pytest
    test = subprocess.run(
        ["pytest", "--maxfail=1", "--disable-warnings", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    test_out = test.stdout.decode()

    report = []
    if errors:
        report.append(f"===== Exceptions since {since} =====\n{errors}\n")
    if test.returncode != 0:
        report.append("===== Pytest Failures =====\n")
        report.append(test_out)

    with open(REPORT, "w") as f:
        f.write("\n".join(report))

    # if anything went wrong, exit non-zero
    if errors or test.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
