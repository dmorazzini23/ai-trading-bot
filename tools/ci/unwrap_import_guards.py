from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = ROOT / 'ai_trading'

def unwrap_import_guards(text: str) -> tuple[str, bool]:
    """Unwrap try/except ImportError guards, converting them to plain imports."""
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)
    changed = False

    def indent_of(s: str) -> int:
        return len(s) - len(s.lstrip(' '))
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith('try:'):
            base_indent = indent_of(line)
            body_start = i + 1
            j = body_start
            try_body_lines = []
            while j < n:
                if lines[j].strip() == '':
                    try_body_lines.append(lines[j])
                    j += 1
                    continue
                if indent_of(lines[j]) <= base_indent:
                    break
                try_body_lines.append(lines[j])
                j += 1
            if j < n and 'except ImportError' in lines[j]:
                except_start = j + 1
                k = except_start
                while k < n:
                    if lines[k].strip() == '':
                        k += 1
                        continue
                    if indent_of(lines[k]) <= base_indent:
                        break
                    k += 1
                except_end = k
                import_statements = []
                non_import_statements = []
                for try_line in try_body_lines:
                    stripped_try = try_line.lstrip()
                    if stripped_try.startswith(('import ', 'from ')) and ' import ' in stripped_try:
                        import_stmt = ' ' * base_indent + stripped_try
                        import_statements.append(import_stmt)
                    elif stripped_try.strip():
                        non_import_statements.append(try_line)
                if import_statements:
                    out.extend(import_statements)
                    if non_import_statements:
                        out.extend(non_import_statements)
                    changed = True
                    i = except_end
                    continue
                else:
                    out.append(line)
                    i += 1
                    continue
            else:
                out.append(line)
                i += 1
                continue
        else:
            out.append(line)
            i += 1
    return (''.join(out), changed)

def main():
    total = 0
    files = 0
    for p in TARGET_ROOT.rglob('*.py'):
        if '/tests/' in p.as_posix():
            continue
        try:
            txt = p.read_text(encoding='utf-8', errors='ignore')
            new, ch = unwrap_import_guards(txt)
            if ch and new != txt:
                try:
                    compile(new, str(p), 'exec')
                    p.write_text(new, encoding='utf-8')
                    total += 1
                except SyntaxError:
                    continue
            files += 1
        except (OSError, PermissionError, KeyError, ValueError, TypeError):
            continue
if __name__ == '__main__':
    main()