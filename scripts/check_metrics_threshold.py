#!/usr/bin/env python
import json
import sys


def main(path: str):
    with open(path) as f:
        metrics = json.load(f)
    if metrics.get('sharpe', 1) <= 0:
        sys.exit('Model performance degraded')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: check_metrics_threshold.py <metrics.json>')
        sys.exit(1)
    main(sys.argv[1])
