#!/usr/bin/env python
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--labels-file", required=True)
    args = parser.parse_args()

    text = (args.title + " " + args.body).lower()
    labels = []
    if "bug" in text or "error" in text:
        labels.append("bug")
    if "feature" in text or "enhancement" in text:
        labels.append("enhancement")

    mapping = {label: {"name": label} for label in labels}
    file_path = args.labels_file
    with open(file_path, "w") as f:
        json.dump(mapping, f)
    print(json.dumps(mapping))


if __name__ == "__main__":
    main()
