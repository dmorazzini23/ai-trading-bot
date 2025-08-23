import logging
import pathlib
import sys
import libcst as cst
import libcst.matchers as m
logger = logging.getLogger(__name__)
PKG = pathlib.Path('ai_trading')
REQ_TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SUBPROC_TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 60

class Fixer(cst.CSTTransformer):
    add_timezone_import = False

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if m.matches(updated_node, m.Call(func=m.Attribute(value=m.Name('requests'), attr=m.OneOf(m.Name('get'), m.Name('post'), m.Name('put'), m.Name('delete'), m.Name('patch'))))):
            if any((m.matches(a, m.Arg(keyword=m.Name('timeout'))) for a in updated_node.args)):
                return updated_node
            return updated_node.with_changes(args=list(updated_node.args) + [cst.Arg(keyword=cst.Name('timeout'), value=cst.Integer(str(REQ_TIMEOUT)))])
        if m.matches(updated_node, m.Call(func=m.Attribute(value=m.Name('subprocess'), attr=m.OneOf(m.Name('run'), m.Name('Popen'), m.Name('call'), m.Name('check_call'), m.Name('check_output'))))):
            kws = {getattr(a.keyword, 'value', None) for a in updated_node.args if a.keyword}
            args = list(updated_node.args)
            if 'timeout' not in kws:
                args.append(cst.Arg(keyword=cst.Name('timeout'), value=cst.Integer(str(SUBPROC_TIMEOUT))))
            if m.matches(updated_node.func, m.Attribute(attr=m.Name('run'))):
                if 'check' not in kws:
                    args.append(cst.Arg(keyword=cst.Name('check'), value=cst.Name('True')))
                if 'capture_output' not in kws:
                    args.append(cst.Arg(keyword=cst.Name('capture_output'), value=cst.Name('True')))
                if 'text' not in kws:
                    args.append(cst.Arg(keyword=cst.Name('text'), value=cst.Name('True')))
            return updated_node.with_changes(args=args)
        if m.matches(updated_node, m.Call(func=m.Attribute(value=m.Name('datetime'), attr=m.Name('now')), args=())):
            self.add_timezone_import = True
            return updated_node.with_changes(args=[cst.Arg(value=cst.Attribute(value=cst.Name('timezone'), attr=cst.Name('utc')))])
        if m.matches(updated_node, m.Call(func=m.Attribute(value=m.Name('yaml'), attr=m.Name('load')))):
            return updated_node.with_changes(func=cst.Attribute(value=cst.Name('yaml'), attr=cst.Name('safe_load')))
        return updated_node

def ensure_timezone_import(mod: cst.Module) -> cst.Module:
    src = mod.code
    if 'from datetime import timezone' in src:
        return mod
    new_import = cst.SimpleStatementLine([cst.ImportFrom(module=cst.Name('datetime'), names=[cst.ImportAlias(cst.Name('timezone'))])])
    return mod.with_changes(body=[new_import] + list(mod.body))

def transform_file(p: pathlib.Path):
    try:
        src = p.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        logger.exception(f'Failed to read file {p}')
        return
    try:
        mod = cst.parse_module(src)
    except (cst.ParserError, ValueError):
        logger.exception(f'Failed to parse file {p}')
        return
    t = Fixer()
    new = mod.visit(t)
    if t.add_timezone_import:
        new = ensure_timezone_import(new)
    if new.code != src:
        p.write_text(new.code, encoding='utf-8')
if __name__ == '__main__':
    for py in PKG.rglob('*.py'):
        transform_file(py)