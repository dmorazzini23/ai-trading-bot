import pathlib
import re

def test_all_requests_have_timeout():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for p in root.rglob("*.py"):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"requests\.(get|post|put|delete|patch)\s*\(", txt):
            # Find the complete function call by matching parentheses
            start = m.start()
            i = start + len(m.group(0)) - 1  # Position of opening parenthesis
            paren_count = 1
            j = i + 1
            while j < len(txt) and paren_count > 0:
                if txt[j] == '(':
                    paren_count += 1
                elif txt[j] == ')':
                    paren_count -= 1
                j += 1
            # Check if timeout appears in the full function call
            full_call = txt[start:j]
            if "timeout=" not in full_call:
                line_no = txt[:start].count('\n') + 1
                first_line = full_call.split('\n')[0]
                offenders.append(f"{p}:{line_no}:{first_line.strip()[:100]}")
    assert not offenders, f"Missing timeout= on requests: {offenders}"