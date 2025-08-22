#!/usr/bin/env python3
"""Codemod script to replace print() with logging and make naive datetime timezone-aware."""

import re
import sys
from pathlib import Path


def replace_print_with_logging(content: str, filepath: str) -> str:
    """Replace print() calls with appropriate logging calls."""
    if "test" in filepath.lower() or filepath.endswith("_test.py"):
        # Skip test files - allow print() in tests as per ruff config
        return content

    # Pattern to match print() calls
    print_pattern = r"\bprint\s*\("

    if not re.search(print_pattern, content):
        return content

    # Check if logging is already imported
    has_logging_import = bool(
        re.search(r"^(import logging|from .*logging.* import)", content, re.MULTILINE)
    )

    # Add logging import at the top if not present
    if not has_logging_import and re.search(print_pattern, content):
        # Find the position after existing imports
        import_lines = []
        other_lines = []
        in_imports = True

        for line in content.split("\n"):
            if in_imports and (
                line.startswith("import ")
                or line.startswith("from ")
                or line.strip() == ""
                or line.startswith("#")
            ):
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)

        # Add logging import
        import_lines.append("import logging")
        import_lines.append("")
        content = "\n".join(import_lines + other_lines)

    # Replace print calls with logger calls
    # Simple print("message") -> logger.info("message")
    content = re.sub(
        r'\bprint\s*\(\s*(["\'][^"\']*["\'])\s*\)', r"logging.info(\1)", content
    )

    # print(f"...") -> logger.info(f"...")
    content = re.sub(
        r'\bprint\s*\(\s*(f["\'][^"\']*["\'])\s*\)', r"logging.info(\1)", content
    )

    # print(variable) -> logger.info(str(variable))
    content = re.sub(
        r"\bprint\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)",
        r"logging.info(str(\1))",
        content,
    )

    # More complex print() calls -> logger.info() with str() wrapper
    content = re.sub(r"\bprint\s*\(([^)]+)\)", r"logging.info(str(\1))", content)

    return content


def replace_datetime_now(content: str) -> str:
    """Replace naive datetime calls with timezone-aware version."""
    # Replace naive datetime calls with timezone-aware equivalents
    content = re.sub(
        r"\bdatetime\.now\s*\(\s*\)", "datetime.now(datetime.timezone.utc)", content
    )

    # Check if we need to import datetime
    if "datetime.now(datetime.timezone.utc)" in content:
        if not re.search(
            r"^(import datetime|from datetime import)", content, re.MULTILINE
        ):
            # Add datetime import
            import_lines = []
            other_lines = []
            in_imports = True

            for line in content.split("\n"):
                if in_imports and (
                    line.startswith("import ")
                    or line.startswith("from ")
                    or line.strip() == ""
                    or line.startswith("#")
                ):
                    import_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)

            import_lines.append("import datetime")
            import_lines.append("")
            content = "\n".join(import_lines + other_lines)

    return content


def process_file(filepath: Path) -> bool:
    """Process a single Python file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            original_content = f.read()

        content = original_content
        content = replace_print_with_logging(content, str(filepath))
        content = replace_datetime_now(content)

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return True

    except Exception:
        return False

    return False


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        target_dirs = [Path(d) for d in sys.argv[1:]]
    else:
        target_dirs = [Path("ai_trading"), Path("scripts")]

    total_modified = 0

    for target_dir in target_dirs:
        if not target_dir.exists():
            continue


        for py_file in target_dir.rglob("*.py"):
            if process_file(py_file):
                total_modified += 1



if __name__ == "__main__":
    main()
