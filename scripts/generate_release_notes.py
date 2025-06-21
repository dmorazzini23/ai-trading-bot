#!/usr/bin/env python
# WARNING: This script is unreferenced in main app/tests. Document its purpose or remove.
import json
import os


def main():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path) as f:
            event = json.load(f)
        release = event.get("release", {})
        tag = release.get("tag_name", "")
        body = release.get("body", "").strip()
    else:
        tag = os.environ.get("GITHUB_REF_NAME", "")
        body = ""
    notes = f"Release {tag}\n\n{body}"
    print(notes)


if __name__ == "__main__":
    main()
