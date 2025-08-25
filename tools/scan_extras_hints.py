#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

DEFAULT_INCLUDE_EXT = {".py", ".md", ".rst", ".txt"}
DEFAULT_EXCLUDES = {
    ".git", "venv", ".venv", "build", "dist", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".eggs", "*.egg-info",
    "node_modules",
}

PKG_TO_EXTRA = {
    "pandas": "pandas",
    "matplotlib": "plot",
    "sklearn": "ml",
    "scikit-learn": "ml",
    "torch": "ml",
    "ta": "ta",
    "ta-lib": "ta",
    "talib": "ta",
}

PKGS_ALT = "|".join(sorted(map(re.escape, PKG_TO_EXTRA.keys()), key=len, reverse=True))
RAW_INSTALL_RE = re.compile(
    rf"""(?ix)
    \b(?:
        pip\s+install\s+({PKGS_ALT})
        |
        install\s+({PKGS_ALT})
    )\b
    """
)


def is_ignored_path(p: Path) -> bool:
    parts = set(p.parts)
    for exc in DEFAULT_EXCLUDES:
        if "*" in exc or "?" in exc:
            if p.match(f"**/{exc}"):
                return True
        elif exc in parts:
            return True
    return False


def line_has_inline_allowlist(line: str) -> bool:
    return ("# extras:ignore" in line) or ("<!-- extras:ignore -->" in line)


def suggest_extra(pkg: str) -> str:
    extra = PKG_TO_EXTRA.get(pkg.lower(), pkg.lower())
    return f'pip install "ai-trading-bot[{extra}]"'


def scan(paths: list[Path]) -> list[tuple[Path, int, int, str, str]]:
    violations = []
    for root in paths:
        root = root.resolve()
        if root.is_file():
            candidates = [root]
        else:
            candidates = [
                p for p in root.rglob("*")
                if p.is_file()
                and not is_ignored_path(p)
                and p.suffix.lower() in DEFAULT_INCLUDE_EXT
            ]
        for f in candidates:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if line_has_inline_allowlist(line):
                    continue
                m = RAW_INSTALL_RE.search(line)
                if not m:
                    continue
                pkg = (m.group(1) or m.group(2) or "").lower()
                if "ai-trading-bot[" in line:
                    continue
                col = m.start() + 1
                snippet = line.strip()
                violations.append((f, i, col, snippet, suggest_extra(pkg)))
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Warn on raw install hints; suggest extras.")
    ap.add_argument("paths", nargs="*", default=["."], help="Paths to scan (default: .)")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on violations.")
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.paths]
    viols = scan(paths)
    if not viols:
        print("[scan_extras_hints] No raw install hints found.")
        return 0
    print(f"[scan_extras_hints] Found {len(viols)} potential raw install hint(s):")
    for path, ln, col, snippet, suggestion in viols:
        print(f"  {path}:{ln}:{col}: {snippet}")
        print(f"    → Suggest: {suggestion}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())

