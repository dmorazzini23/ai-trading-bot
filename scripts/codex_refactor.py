#!/usr/bin/env python
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Codex refactor placeholder")
    parser.add_argument("--diff", help="diff to refactor", default="")
    args = parser.parse_args()

    # Simulate generating a refactored file so the workflow has something to commit
    suggestions_dir = Path("refactor-suggestions")
    suggestions_dir.mkdir(exist_ok=True)
    output_file = suggestions_dir / "refactored_example.py"
    output_file.write_text(
        "# Auto-generated refactor suggestion\n\n"
        + f"# Received diff length: {len(args.diff)}\n"
    )
    print(f"Wrote refactor suggestion to {output_file}")


if __name__ == "__main__":
    main()
