import pathlib
import sys

import libcst as cst
import libcst.matchers as m


class FixNoneComparisons(cst.CSTTransformer):
    def leave_Comparison(self, original_node, updated_node):
        ops = []
        for op in updated_node.comparisons:
            left_none  = m.matches(updated_node.left, m.Name("None"))
            right_none = m.matches(op.comparator, m.Name("None"))
            if isinstance(op.operator, cst.Equal) and (left_none or right_none):
                ops.append(op.with_changes(operator=cst.Is()))
            elif isinstance(op.operator, cst.NotEqual) and (left_none or right_none):
                ops.append(op.with_changes(operator=cst.IsNot()))
            else:
                ops.append(op)
        return updated_node.with_changes(comparisons=ops)

def run(root="."):
    for p in pathlib.Path(root).rglob("*.py"):
        if any(part in {".venv","dist","build","artifacts",".tox"} for part in p.parts):
            continue
        try:
            src = p.read_text(encoding="utf-8")
            mod = cst.parse_module(src)
            out = mod.visit(FixNoneComparisons())
            if out.code != src:
                p.write_text(out.code, encoding="utf-8")
                print(f"REWROTE {p}")
        # noqa: BLE001 TODO: narrow exception
        except Exception as e:
            print(f"SKIP {p}: {e}", file=sys.stderr)

if __name__ == "__main__":
    run()
