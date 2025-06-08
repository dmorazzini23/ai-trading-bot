#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser(description="Codex refactor placeholder")
    parser.add_argument("--diff", help="diff to refactor", default="")
    args = parser.parse_args()
    print("Codex refactor placeholder; diff length:", len(args.diff))


if __name__ == "__main__":
    main()
