from __future__ import annotations
from pathlib import Path
import libcst as cst
ROOT = Path('ai_trading')

class StripImportGuards(cst.CSTTransformer):

    def leave_Try(self, original: cst.Try, updated: cst.Try) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
        if not updated.handlers:
            return updated
        if not all((isinstance(h.type, cst.Name) and h.type.value == 'ImportError' for h in updated.handlers if h.type)):
            return updated
        body = updated.body
        if not isinstance(body, cst.IndentedBlock):
            return updated
        import_statements = []
        for stmt in body.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for substmt in stmt.body:
                    if isinstance(substmt, cst.Import | cst.ImportFrom):
                        import_statements.append(stmt)
                        break
                else:
                    return updated
            else:
                return updated
        if import_statements:
            return cst.FlattenSentinel(import_statements)
        else:
            return updated

def transform_file(p: Path) -> bool:
    try:
        src = p.read_text(encoding='utf-8')
        mod = cst.parse_module(src)
        new = mod.visit(StripImportGuards())
        if new.code != src:
            p.write_text(new.code, encoding='utf-8')
            return True
    except (OSError, PermissionError, KeyError, ValueError, TypeError):
        pass
    return False

def main():
    changed = 0
    for py in ROOT.rglob('*.py'):
        if transform_file(py):
            changed += 1
if __name__ == '__main__':
    main()