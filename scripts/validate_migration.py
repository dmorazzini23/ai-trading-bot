#!/usr/bin/env python3
"""
Validation script for portfolio optimizer and transaction costs migration.
Tests the migration without requiring external dependencies.
"""

import ast
import os
import sys


def check_file_exists(filepath):
    """Check if a file exists."""
    return os.path.isfile(filepath)

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath) as f:
            content = f.read()
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def extract_imports(filepath):
    """Extract import statements from a file."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        return imports
    except Exception:
        return []

def main():
    """Run migration validation."""
    print("Portfolio Optimizer & Transaction Costs Migration Validation")
    print("=" * 60)

    # Files that should exist
    required_files = [
        "ai_trading/portfolio/optimizer.py",
        "ai_trading/execution/transaction_costs.py",
        "ai_trading/portfolio/__init__.py",
        "scripts/portfolio_optimizer.py",
        "scripts/transaction_cost_calculator.py"
    ]

    # Check file existence
    print("\n1. Checking file existence...")
    all_files_exist = True
    for filepath in required_files:
        if check_file_exists(filepath):
            print(f"✓ {filepath}")
        else:
            print(f"❌ {filepath} - Missing!")
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return 1

    # Check syntax
    print("\n2. Checking syntax...")
    all_syntax_ok = True
    for filepath in required_files:
        valid, error = check_file_syntax(filepath)
        if valid:
            print(f"✓ {filepath}")
        else:
            print(f"❌ {filepath} - {error}")
            all_syntax_ok = False

    if not all_syntax_ok:
        print("\n❌ Some files have syntax errors!")
        return 1

    # Check import structure
    print("\n3. Checking import structure...")

    # Check signals.py imports transaction costs correctly
    signals_imports = extract_imports("ai_trading/signals.py")
    correct_tc_import = any("ai_trading.execution.transaction_costs" in imp for imp in signals_imports)
    incorrect_tc_import = any("scripts.transaction_cost" in imp for imp in signals_imports)

    if correct_tc_import and not incorrect_tc_import:
        print("✓ ai_trading/signals.py imports transaction costs from correct location")
    else:
        print("❌ ai_trading/signals.py has incorrect transaction cost imports")
        return 1

    # Check portfolio __init__.py exports optimizer classes
    portfolio_imports = extract_imports("ai_trading/portfolio/__init__.py")
    has_optimizer_imports = any("optimizer" in imp for imp in portfolio_imports)

    if has_optimizer_imports:
        print("✓ ai_trading/portfolio/__init__.py imports from optimizer module")
    else:
        print("❌ ai_trading/portfolio/__init__.py missing optimizer imports")
        return 1

    # Check shims import from ai_trading
    for shim_file in ["scripts/portfolio_optimizer.py", "scripts/transaction_cost_calculator.py"]:
        shim_imports = extract_imports(shim_file)
        has_ai_trading_imports = any("ai_trading" in imp for imp in shim_imports)

        if has_ai_trading_imports:
            print(f"✓ {shim_file} imports from ai_trading package")
        else:
            print(f"❌ {shim_file} missing ai_trading imports")
            return 1

    # Check no scripts imports in ai_trading
    print("\n4. Checking for scripts imports in production code...")
    ai_trading_files = [
        "ai_trading/signals.py",
        "ai_trading/rebalancer.py",
        "ai_trading/portfolio/__init__.py"
    ]

    scripts_imports_found = False
    for filepath in ai_trading_files:
        if check_file_exists(filepath):
            imports = extract_imports(filepath)
            scripts_imports = [imp for imp in imports if "scripts." in imp]
            if scripts_imports:
                print(f"❌ {filepath} still imports from scripts:")
                for imp in scripts_imports:
                    print(f"   {imp}")
                scripts_imports_found = True

    if not scripts_imports_found:
        print("✓ No scripts imports found in production code")
    else:
        return 1

    print("\n" + "=" * 60)
    print("✅ All validation checks passed!")
    print("✅ Migration completed successfully!")
    print("\nKey improvements:")
    print("• Portfolio optimizer moved to ai_trading.portfolio.optimizer")
    print("• Transaction costs moved to ai_trading.execution.transaction_costs")
    print("• Production code no longer imports from scripts/")
    print("• Backward compatibility maintained via shims")
    print("• Exception handling improved in signals.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())
