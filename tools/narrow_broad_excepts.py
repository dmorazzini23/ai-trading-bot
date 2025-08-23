"""Script to narrow broad exception handlers across the repository."""
from __future__ import annotations
import ast
import io
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from ai_trading.logging import get_logger
logger = get_logger(__name__)

class Narrower(ast.NodeTransformer):
    """AST transformer that replaces ``except Exception`` with specific tuples."""

    def __init__(self, src: str, imports: set[str]):
        self.src = src
        self.imports = imports
        self._stack: list[ast.AST] = []
        super().__init__()

    def visit(self, node):
        self._stack.append(node)
        out = super().visit(node)
        self._stack.pop()
        return out

    def inside_main_guard(self) -> bool:
        """Return ``True`` when within an ``if __name__ == '__main__'`` block."""
        for n in self._stack:
            if isinstance(n, ast.If):
                if hasattr(ast, 'unparse'):
                    test = ast.unparse(n.test)
                else:
                    buf = io.StringIO()
                    ast.dump(n.test, file=buf)
                    test = buf.getvalue()
                if '__name__' in test and "'__main__'" in test:
                    return True
        return False

    def _exception_tuple_for(self, try_node: ast.Try) -> list[ast.expr]:
        imps = self.imports
        excs: list[ast.expr] = []

        def name(id_: str) -> ast.expr:
            return ast.Name(id=id_, ctx=ast.Load())
        src_block = ast.get_source_segment(self.src, try_node) or ''
        if any((x in imps for x in {'pandas', 'pd'})):
            excs.append(ast.Attribute(value=ast.Attribute(value=name('pd'), attr='errors', ctx=ast.Load()), attr='EmptyDataError', ctx=ast.Load()))
            excs.extend([name('KeyError'), name('ValueError'), name('TypeError')])
        if 'json' in imps:
            excs.append(ast.Attribute(value=name('json'), attr='JSONDecodeError', ctx=ast.Load()))
            excs.extend([name('ValueError'), name('OSError')])
        if 'requests' in imps:
            for a in ('Timeout', 'ConnectionError', 'HTTPError', 'RequestException'):
                excs.append(ast.Attribute(value=name('requests'), attr=a, ctx=ast.Load()))
            excs.extend([name('ValueError'), name('KeyError')])
        if 'subprocess' in imps or 'from subprocess' in self.src:
            for a in ('CalledProcessError', 'TimeoutExpired'):
                excs.append(ast.Attribute(value=name('subprocess'), attr=a, ctx=ast.Load()))
            excs.extend([name('FileNotFoundError'), name('OSError')])
        if 'signal' in imps:
            excs.extend([name('ValueError'), name('OSError')])
        if any((tok in src_block for tok in ('open(', '.read_', '.to_', 'Path(', '.write_', '.exists(', '.mkdir(', '.unlink(', '.rename('))):
            excs.extend([name('OSError'), name('PermissionError')])
        if any((tok in self.src for tok in ('numpy', 'np', 'math'))):
            excs.extend([name('ValueError'), name('TypeError'), name('ZeroDivisionError'), name('OverflowError')])
        excs.extend([name('KeyError'), name('ValueError'), name('TypeError')])
        uniq: list[ast.expr] = []
        seen = set()
        for e in excs:
            dumped = ast.dump(e)
            if dumped not in seen:
                seen.add(dumped)
                uniq.append(e)
        return uniq

    def visit_Try(self, node: ast.Try):
        if self.inside_main_guard():
            return self.generic_visit(node)
        for h in node.handlers:
            if isinstance(h.type, ast.Name) and h.type.id == 'Exception':
                h.type = ast.Tuple(elts=self._exception_tuple_for(node), ctx=ast.Load())
        return self.generic_visit(node)

def collect_imports(tree: ast.AST, src: str) -> set[str]:
    imps: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imps.add(a.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imps.add(n.module.split('.')[0])
    if 'import pandas as pd' in src:
        imps.add('pd')
    return imps

def rewrite_file(path: Path) -> bool:
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    imps = collect_imports(tree, src)
    new_tree = Narrower(src, imps).visit(tree)
    ast.fix_missing_locations(new_tree)
    new_src = ast.unparse(new_tree)
    if new_src != src:
        path.write_text(new_src, encoding='utf-8')
        return True
    return False

def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    changed: list[str] = []
    for p in root.rglob('*.py'):
        s = str(p)
        if 'tests' in p.parts:
            continue
        if any((seg in s for seg in ('ai_trading/core/bot_engine.py', 'ai_trading/runner.py', 'trade_execution/__init__.py'))):
            continue
        if any((seg in p.parts for seg in ('scripts', 'tools', 'ai_trading'))):
            if rewrite_file(p):
                changed.append(s)
    logger.info('NARROWED_EXCEPT_FILES', extra={'count': len(changed)})
    for c in changed:
        logger.info('NARROWED_FILE', extra={'file': c})
    return 0
if __name__ == '__main__':
    raise SystemExit(main())