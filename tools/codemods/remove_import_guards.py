from __future__ import annotations

from pathlib import Path

import libcst as cst

ROOT = Path("ai_trading")

class StripImportGuards(cst.CSTTransformer):
    def leave_Try(self, original: cst.Try, updated: cst.Try) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
        # If the try-body is ONLY import statements and except ImportError exists,
        # replace the whole try/except with just the import statements.
        if not updated.handlers:
            return updated

        # Check if all handlers are ImportError
        if not all(
            isinstance(h.type, cst.Name) and h.type.value == "ImportError"
            for h in updated.handlers if h.type
        ):
            return updated

        body = updated.body
        if not isinstance(body, cst.IndentedBlock):
            return updated

        # Check if all statements in try block are import statements
        import_statements = []
        for stmt in body.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for substmt in stmt.body:
                    if isinstance(substmt, cst.Import | cst.ImportFrom):
                        import_statements.append(stmt)
                        break
                else:
                    # Non-import statement found
                    return updated
            else:
                # Non-simple statement found
                return updated

        # Return only the import statements (flatten)
        if import_statements:
            return cst.FlattenSentinel(import_statements)
        else:
            return updated

def transform_file(p: Path) -> bool:
    try:
        src = p.read_text(encoding="utf-8")
        mod = cst.parse_module(src)
        new = mod.visit(StripImportGuards())
        if new.code != src:
            p.write_text(new.code, encoding="utf-8")
            return True
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        pass
    return False

def main():
    changed = 0
    for py in ROOT.rglob("*.py"):
        if transform_file(py):
            changed += 1

if __name__ == "__main__":
    main()
