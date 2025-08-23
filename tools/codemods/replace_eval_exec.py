from __future__ import annotations
import re
from pathlib import Path
ROOT = Path('ai_trading')
DISPATCH_HINT = '# TODO: replace dynamic eval/exec with explicit dispatch\n# Example:\n# DISPATCH = {\'momentum\': run_momentum, \'mean_reversion\': run_mean_reversion}\n# fn = DISPATCH.get(name) or (_ for _ in ()).throw(ValueError(f"Unknown strategy {name}"))\n# return fn(*args, **kwargs)\n'

def transform_text(text: str) -> tuple[str, bool]:
    out = []
    lines = text.splitlines()
    i = 0
    changed = False
    while i < len(lines):
        L = lines[i]
        if re.search('\\bexec\\s*\\(', L) or re.search('(^|[^.])\\beval\\s*\\(', L):
            if '.eval(' in L or 'model.eval()' in L or '_MODEL.eval()' in L:
                out.append(L)
            else:
                out.append(DISPATCH_HINT)
                out.append('# ' + L)
                changed = True
        else:
            out.append(L)
        i += 1
    return ('\n'.join(out), changed)

def main():
    changed_files = 0
    for p in ROOT.rglob('*.py'):
        try:
            t = p.read_text(encoding='utf-8', errors='ignore')
            new, changed = transform_text(t)
            if changed:
                p.write_text(new, encoding='utf-8')
                changed_files += 1
        except (OSError, PermissionError, KeyError, ValueError, TypeError):
            pass
if __name__ == '__main__':
    main()