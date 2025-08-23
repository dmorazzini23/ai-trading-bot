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
        return (True, '')
    except SyntaxError as e:
        return (False, f'Syntax error: {e}')
    except (OSError, PermissionError, KeyError, ValueError, TypeError) as e:
        return (False, f'Error: {e}')

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
                    imports.append(f'from {module} import {alias.name}')
        return imports
    except (OSError, PermissionError, KeyError, ValueError, TypeError):
        return []

def main():
    """Run migration validation."""
    required_files = ['ai_trading/portfolio/optimizer.py', 'ai_trading/execution/transaction_costs.py', 'ai_trading/portfolio/__init__.py', 'scripts/portfolio_optimizer.py', 'scripts/transaction_cost_calculator.py']
    all_files_exist = True
    for filepath in required_files:
        if check_file_exists(filepath):
            pass
        else:
            all_files_exist = False
    if not all_files_exist:
        return 1
    all_syntax_ok = True
    for filepath in required_files:
        valid, error = check_file_syntax(filepath)
        if valid:
            pass
        else:
            all_syntax_ok = False
    if not all_syntax_ok:
        return 1
    signals_imports = extract_imports('ai_trading/signals.py')
    correct_tc_import = any(('ai_trading.execution.transaction_costs' in imp for imp in signals_imports))
    incorrect_tc_import = any(('scripts.transaction_cost' in imp for imp in signals_imports))
    if correct_tc_import and (not incorrect_tc_import):
        pass
    else:
        return 1
    portfolio_imports = extract_imports('ai_trading/portfolio/__init__.py')
    has_optimizer_imports = any(('optimizer' in imp for imp in portfolio_imports))
    if has_optimizer_imports:
        pass
    else:
        return 1
    for shim_file in ['scripts/portfolio_optimizer.py', 'scripts/transaction_cost_calculator.py']:
        shim_imports = extract_imports(shim_file)
        has_ai_trading_imports = any(('ai_trading' in imp for imp in shim_imports))
        if has_ai_trading_imports:
            pass
        else:
            return 1
    ai_trading_files = ['ai_trading/signals.py', 'ai_trading/rebalancer.py', 'ai_trading/portfolio/__init__.py']
    scripts_imports_found = False
    for filepath in ai_trading_files:
        if check_file_exists(filepath):
            imports = extract_imports(filepath)
            scripts_imports = [imp for imp in imports if 'scripts.' in imp]
            if scripts_imports:
                for imp in scripts_imports:
                    pass
                scripts_imports_found = True
    if not scripts_imports_found:
        pass
    else:
        return 1
    return 0
if __name__ == '__main__':
    sys.exit(main())