from __future__ import annotations
import os, sys, subprocess, py_compile

# AI-AGENT-REF: compile only tracked Python files

def git_ls_files() -> list[str]:
    """Return tracked Python files or empty list on failure."""  # AI-AGENT-REF: avoid .git traversal
    try:
        out = subprocess.check_output(["git", "ls-files", "*.py"], text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []

def fallback_walk(root: str) -> list[str]:
    """Fallback tree walk excluding common non-source dirs."""
    skip = {".git", "__pycache__", "venv", ".venv", ".tox"}
    paths: list[str] = []
    for d, dirs, files in os.walk(root):
        if os.path.basename(d) in skip:
            dirs[:] = []
            continue
        for f in files:
            if f.endswith(".py"):
                paths.append(os.path.join(d, f))
    return paths

def main() -> int:
    files = git_ls_files() or fallback_walk(".")
    for p in files:
        try:
            py_compile.compile(p, doraise=True)
        except py_compile.PyCompileError as e:
            sys.stderr.write(str(e) + "\n")
            return 1
    print("py_compile OK (git-aware)")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
