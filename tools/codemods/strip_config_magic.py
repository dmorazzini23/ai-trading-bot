from __future__ import annotations
from pathlib import Path
import libcst as cst
import libcst.matchers as m

CFG_INIT = Path("ai_trading/config/__init__.py")

class StripConfigMagic(cst.CSTTransformer):
    def leave_FunctionDef(self, original: cst.FunctionDef, updated: cst.FunctionDef):
        if original.name.value == "__getattr__":
            return cst.RemovalSentinel.REMOVE
        return updated

def remove_uppercase_properties(settings_path: Path) -> None:
    if not settings_path.exists():
        return
        
    src = settings_path.read_text(encoding="utf-8")
    mod = cst.parse_module(src)

    class KillUpperProps(cst.CSTTransformer):
        def leave_FunctionDef(self, o: cst.FunctionDef, u: cst.FunctionDef):
            # Check if this is a property decorator with uppercase name
            has_property_decorator = False
            for decorator in o.decorators or []:
                if m.matches(decorator, m.Decorator(decorator=m.Name("property"))):
                    has_property_decorator = True
                    break
            
            if has_property_decorator and o.name.value.isupper():
                return cst.RemovalSentinel.REMOVE
            return u

    new = mod.visit(KillUpperProps())
    if new.code != src:
        settings_path.write_text(new.code, encoding="utf-8")
        print(f"Removed uppercase properties from: {settings_path}")

def main():
    if CFG_INIT.exists():
        try:
            src = CFG_INIT.read_text(encoding="utf-8")
            new = cst.parse_module(src).visit(StripConfigMagic())
            if new.code != src:
                CFG_INIT.write_text(new.code, encoding="utf-8")
                print(f"Removed __getattr__ from: {CFG_INIT}")
        except Exception as e:
            print(f"Error processing {CFG_INIT}: {e}")
    
    settings = Path("ai_trading/config/settings.py")
    remove_uppercase_properties(settings)
    
    # Also check management.py for __getattr__
    management = Path("ai_trading/config/management.py")
    if management.exists():
        try:
            src = management.read_text(encoding="utf-8")
            new = cst.parse_module(src).visit(StripConfigMagic())
            if new.code != src:
                management.write_text(new.code, encoding="utf-8")
                print(f"Removed __getattr__ from: {management}")
        except Exception as e:
            print(f"Error processing {management}: {e}")

if __name__ == "__main__":
    main()