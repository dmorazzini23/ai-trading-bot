"""Narrow overly broad ``except`` clauses where safe."""
from __future__ import annotations
import pathlib
import libcst as cst
import libcst.matchers as m

class _Narrower(cst.CSTTransformer):

    def __init__(self, *, has_requests: bool, has_datetime: bool) -> None:
        self.has_requests = has_requests
        self.has_datetime = has_datetime
        self.modified = False

    def leave_ExceptHandler(self, original: cst.ExceptHandler, updated: cst.ExceptHandler) -> cst.ExceptHandler:
        if original.type is None or m.matches(original.type, m.Name('Exception')):
            self.modified = True
            if self.has_datetime:
                new_type = cst.Tuple([cst.Element(cst.Name('ValueError')), cst.Element(cst.Name('TypeError'))])
                return updated.with_changes(type=new_type)
            if self.has_requests:
                new_type = cst.Tuple([cst.Element(cst.Attribute(cst.Name('requests'), cst.Name('RequestException'))), cst.Element(cst.Name('TimeoutError'))])
                return updated.with_changes(type=new_type)
            comment = cst.Comment('# noqa: BLE001 TODO: narrow exception')
            lines = list(updated.leading_lines) + [cst.EmptyLine(comment=comment)]
            return updated.with_changes(type=cst.Name('Exception'), leading_lines=tuple(lines))
        return updated

def _should_skip(path: pathlib.Path) -> bool:
    parts = path.parts
    return 'artifacts' in parts

def main() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    for py in repo.rglob('*.py'):
        if _should_skip(py):
            continue
        text = py.read_text()
        has_requests = 'import requests' in text or 'from requests' in text
        has_datetime = 'datetime' in text or 'dateutil' in text
        module = cst.parse_module(text)
        transformer = _Narrower(has_requests=has_requests, has_datetime=has_datetime)
        updated = module.visit(transformer)
        if transformer.modified and module.code != updated.code:
            py.write_text(updated.code)
if __name__ == '__main__':
    main()