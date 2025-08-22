from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / "ai_trading"

def unwrap_import_guards(text: str) -> tuple[str, bool]:
    """Unwrap try/except ImportError guards, converting them to plain imports."""
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)
    changed = False

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    while i < n:
        line = lines[i]
        stripped = line.lstrip()

        # Look for try: statements
        if stripped.startswith("try:"):
            base_indent = indent_of(line)
            body_start = i + 1

            # Collect try-body (strictly deeper indent)
            j = body_start
            try_body_lines = []

            while j < n:
                if lines[j].strip() == "":
                    try_body_lines.append(lines[j])
                    j += 1
                    continue
                if indent_of(lines[j]) <= base_indent:
                    break
                try_body_lines.append(lines[j])
                j += 1
            body_end = j

            # Check if next block is "except ImportError"
            if j < n and "except ImportError" in lines[j]:
                except_start = j + 1
                k = except_start

                # Skip the except block
                while k < n:
                    if lines[k].strip() == "":
                        k += 1
                        continue
                    if indent_of(lines[k]) <= base_indent:
                        break
                    k += 1
                except_end = k

                # Extract ONLY import statements from try body
                import_statements = []
                non_import_statements = []

                for try_line in try_body_lines:
                    stripped_try = try_line.lstrip()
                    if stripped_try.startswith(("import ", "from ")) and " import " in stripped_try:
                        # This is an import - extract it and dedent to base level
                        import_stmt = " " * base_indent + stripped_try
                        import_statements.append(import_stmt)
                    elif stripped_try.strip():  # Non-empty, non-import line
                        non_import_statements.append(try_line)

                # If we found import statements, unwrap them
                if import_statements:
                    # Add the unwrapped imports
                    out.extend(import_statements)

                    # If there were non-import statements, preserve them outside the try/except
                    if non_import_statements:
                        out.extend(non_import_statements)

                    changed = True
                    # Skip the entire try/except block
                    i = except_end
                    continue
                else:
                    # No imports found - preserve the original try/except
                    out.append(line)
                    i += 1
                    continue
            else:
                # Not an ImportError exception - preserve as is
                out.append(line)
                i += 1
                continue
        else:
            # Not a try statement - preserve as is
            out.append(line)
            i += 1

    return ("".join(out), changed)

def main():
    total = 0
    files = 0
    for p in TARGET_ROOT.rglob("*.py"):
        # Keep tests out of this pass
        if "/tests/" in p.as_posix():
            continue

        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            new, ch = unwrap_import_guards(txt)
            if ch and new != txt:
                # Verify the new code is syntactically valid before writing
                try:
                    compile(new, str(p), 'exec')
                    p.write_text(new, encoding="utf-8")
                    total += 1
                    print(f"Successfully unwrapped imports in: {p.relative_to(ROOT)}")
                except SyntaxError as e:
                    print(f"Syntax error would be introduced in {p.relative_to(ROOT)}, skipping: {e}")
                    continue
            files += 1
        except Exception as e:
            print(f"Error processing {p.relative_to(ROOT)}: {e}")
            continue

    print(f"files_scanned={files} files_modified={total}")

if __name__ == "__main__":
    main()
