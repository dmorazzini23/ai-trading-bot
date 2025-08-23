"""Codemod script to replace print() with logging and make naive datetime timezone-aware."""
import re
import sys
from pathlib import Path

def replace_print_with_logging(content: str, filepath: str) -> str:
    """Replace print() calls with appropriate logging calls."""
    if 'test' in filepath.lower() or filepath.endswith('_test.py'):
        return content
    print_pattern = '\\bprint\\s*\\('
    if not re.search(print_pattern, content):
        return content
    has_logging_import = bool(re.search('^(import logging|from .*logging.* import)', content, re.MULTILINE))
    if not has_logging_import and re.search(print_pattern, content):
        import_lines = []
        other_lines = []
        in_imports = True
        for line in content.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        import_lines.append('import logging')
        import_lines.append('')
        content = '\n'.join(import_lines + other_lines)
    content = re.sub('\\bprint\\s*\\(\\s*(["\\\'][^"\\\']*["\\\'])\\s*\\)', 'logging.info(\\1)', content)
    content = re.sub('\\bprint\\s*\\(\\s*(f["\\\'][^"\\\']*["\\\'])\\s*\\)', 'logging.info(\\1)', content)
    content = re.sub('\\bprint\\s*\\(\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\)', 'logging.info(str(\\1))', content)
    content = re.sub('\\bprint\\s*\\(([^)]+)\\)', 'logging.info(str(\\1))', content)
    return content

def replace_datetime_now(content: str) -> str:
    """Replace naive datetime calls with timezone-aware version."""
    content = re.sub('\\bdatetime\\.now\\s*\\(\\s*\\)', 'datetime.now(datetime.timezone.utc)', content)
    if 'datetime.now(datetime.timezone.utc)' in content:
        if not re.search('^(import datetime|from datetime import)', content, re.MULTILINE):
            import_lines = []
            other_lines = []
            in_imports = True
            for line in content.split('\n'):
                if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
                    import_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)
            import_lines.append('import datetime')
            import_lines.append('')
            content = '\n'.join(import_lines + other_lines)
    return content

def process_file(filepath: Path) -> bool:
    """Process a single Python file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            original_content = f.read()
        content = original_content
        content = replace_print_with_logging(content, str(filepath))
        content = replace_datetime_now(content)
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except (ValueError, TypeError):
        return False
    return False

def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        target_dirs = [Path(d) for d in sys.argv[1:]]
    else:
        target_dirs = [Path('ai_trading'), Path('scripts')]
    total_modified = 0
    for target_dir in target_dirs:
        if not target_dir.exists():
            continue
        for py_file in target_dir.rglob('*.py'):
            if process_file(py_file):
                total_modified += 1
if __name__ == '__main__':
    main()