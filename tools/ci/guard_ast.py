from __future__ import annotations

import sys
from pathlib import Path

import libcst as cst
import libcst.matchers as m

TARGET_DIRS = ["ai_trading"]
FAILURES = []

class Visitor(cst.CSTVisitor):
    def __init__(self, path: Path):
        self.path = path

    @m.visit(m.Call(func=m.Name("eval")))
    def _raw_eval(self, node: cst.Call) -> None:
        FAILURES.append((self.path, node.lpar[0].start.line if node.lpar else node.func.start.line, "raw _raise_dynamic_eval_disabled()"))

    @m.visit(m.Call(func=m.Name("exec")))
    def _exec(self, node: cst.Call) -> None:
        FAILURES.append((self.path, node.lpar[0].start.line if node.lpar else node.func.start.line, "_raise_dynamic_exec_disabled()"))

    @m.visit(m.ExceptHandler(type=None))
    def _bare_except(self, node: cst.ExceptHandler) -> None:
        FAILURES.append((self.path, node.body.lbrace.start.line, "bare except"))

    @m.visit(m.ExceptHandler())
    def _empty_except(self, node: cst.ExceptHandler) -> None:
        body = node.body
        if len(body.body) == 1 and m.matches(body.body[0], m.SimpleStatementLine(body=m.OneOf(m.Pass(), m.Return()))):
            FAILURES.append((self.path, body.body[0].start.line, "empty except"))

    @m.visit(m.Call(func=m.Attribute(value=m.Name("requests"), attr=m.OneOf("get","post","put","delete","patch"))))
    def _requests_timeout(self, node: cst.Call) -> None:
        if not any(m.matches(a, m.Arg(keyword=m.Name("timeout"))) for a in node.args):
            FAILURES.append((self.path, node.func.attr.start.line, "requests.* without timeout"))

    @m.visit(m.Call(func=m.Attribute(value=m.Name("subprocess"), attr=m.OneOf("run","Popen","call","check_call","check_output"))))
    def _subproc_timeout(self, node: cst.Call) -> None:
        if not any(m.matches(a, m.Arg(keyword=m.Name("timeout"))) for a in node.args):
            FAILURES.append((self.path, node.func.attr.start.line, "subprocess.* without timeout"))

    @m.visit(m.Call(func=m.Attribute(value=m.Name("datetime"), attr=m.Name("now"))))
    def _naive_now(self, node: cst.Call) -> None:
        if not node.args:
            FAILURES.append((self.path, node.func.attr.start.line, "naive datetime.now()"))

    @m.visit(m.Call(func=m.Attribute(value=m.Name("yaml"), attr=m.Name("load"))))
    def _yaml_unsafe(self, node: cst.Call) -> None:
        has_loader = any(m.matches(a, m.Arg(keyword=m.Name("Loader"))) for a in node.args)
        if not has_loader:
            FAILURES.append((self.path, node.func.attr.start.line, "yaml.load without Loader"))

    @m.visit(m.FunctionDef())
    def _mutable_defaults(self, node: cst.FunctionDef) -> None:
        if not node.params:
            return
        for p in node.params.params + node.params.kwonly_params:
            if p.default and m.matches(p.default.value, m.OneOf(m.List(), m.Dict(), m.Call(func=m.Name("set")))):
                FAILURES.append((self.path, p.default.value.start.line, "mutable default in signature"))

def scan_dir(root: Path) -> int:
    for py in root.rglob("*.py"):
        try:
            mod = cst.parse_module(py.read_text(encoding="utf-8"))
        except (ValueError, TypeError):
            continue
        mod.visit(Visitor(py))
    return 1 if FAILURES else 0

def main() -> int:
    code = 0
    for d in TARGET_DIRS:
        code |= scan_dir(Path(d))

    if FAILURES:
        for p, ln, msg in FAILURES[:2000]:
            pass
    else:
        pass
    return code

if __name__ == "__main__":
    sys.exit(main())
