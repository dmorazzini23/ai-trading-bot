#!/usr/bin/env python
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()

    text = (args.title + " " + args.body).lower()
    labels = []
    if "bug" in text or "error" in text:
        labels.append("bug")
    if "feature" in text or "enhancement" in text:
        labels.append("enhancement")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(labels, f)
    print(" ".join(labels))


if __name__ == "__main__":
    main()
