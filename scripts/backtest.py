#!/usr/bin/env python
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='metrics.json')
    parser.add_argument('--plot', default='metrics.png')
    args = parser.parse_args()

    metrics = {"sharpe": 1.0, "net_pnl": 0.0}
    with open(args.output, 'w') as f:
        json.dump(metrics, f)

    with open(args.plot, 'wb') as f:
        f.write(b'')
    print('Dummy backtest complete')


if __name__ == '__main__':
    main()
