"""Codemod script for IO safety, YAML hardening, async sleep fixes, import cleanup, and path literals."""
import re
import sys
from pathlib import Path

def replace_yaml_load(content: str) -> str:
    """Replace unsafe YAML loading with yaml.safe_load()."""
    content = re.sub('yaml\\.load\\s*\\(\\s*([^)]+)\\s*\\)', lambda m: f'yaml.safe_load({m.group(1).strip()})' if 'Loader=' not in m.group(1) else m.group(0), content)
    return content

def add_requests_timeout(content: str) -> str:
    """Add timeout= parameter to requests.* calls."""
    requests_pattern = 'requests\\.(get|post|put|delete|patch)\\s*\\('
    lines = content.split('\n')
    modified_lines = []
    for i, line in enumerate(lines):
        if re.search(requests_pattern, line):
            has_timeout = False
            check_lines = lines[i:i + 6]
            for check_line in check_lines:
                if 'timeout=' in check_line:
                    has_timeout = True
                    break
            if not has_timeout:
                if ')' in line:
                    line = re.sub('(requests\\.(get|post|put|delete|patch)\\s*\\([^)]*)\\)', '\\1, timeout=30)', line)
                else:
                    line = re.sub('(requests\\.(get|post|put|delete|patch)\\s*\\()', '\\1timeout=30, ', line)
        modified_lines.append(line)
    return '\n'.join(modified_lines)

def harden_subprocess(content: str) -> str:
    """Add safety parameters to subprocess calls."""
    subprocess_pattern = 'subprocess\\.(run|Popen|call|check_call|check_output)\\s*\\('
    lines = content.split('\n')
    modified_lines = []
    for i, line in enumerate(lines):
        if re.search(subprocess_pattern, line):
            has_timeout = False
            check_lines = lines[i:i + 6]
            for check_line in check_lines:
                if 'timeout=' in check_line:
                    has_timeout = True
                if 'shell=False' in check_line:
                    pass
            if not has_timeout:
                if ')' in line:
                    line = re.sub('(subprocess\\.(run|Popen|call|check_call|check_output)\\s*\\([^)]*)\\)', '\\1, timeout=30)', line)
                else:
                    line = re.sub('(subprocess\\.(run|Popen|call|check_call|check_output)\\s*\\()', '\\1timeout=30, ', line)
        modified_lines.append(line)
    return '\n'.join(modified_lines)

def replace_wildcard_imports(content: str) -> str:
    """Replace wildcard imports with explicit imports."""
    lines = content.split('\n')
    modified_lines = []
    for line in lines:
        if re.match('from\\s+\\.\\w+\\s+import\\s+\\*', line) and '__all__' not in content:
            modified_lines.append(f'# TODO: Replace wildcard import: {line}')
            modified_lines.append(f'# {line}')
        else:
            modified_lines.append(line)
    return '\n'.join(modified_lines)

def replace_data_paths(content: str) -> str:
    """Replace 'data/' path literals with paths.data_dir()."""
    data_path_pattern = '["\\\'](\\./)?data/([^"\\\']*)["\\\']'

    def replace_match(match):
        subpath = match.group(2)
        if subpath:
            return f'paths.data_dir() / "{subpath}"'
        else:
            return 'paths.data_dir()'
    content = re.sub(data_path_pattern, replace_match, content)
    if 'paths.data_dir()' in content:
        if not re.search('from .*paths.* import|import.*paths', content):
            import_lines = []
            other_lines = []
            in_imports = True
            for line in content.split('\n'):
                if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
                    import_lines.append(line)
                else:
                    in_imports = False
                    other_lines.append(line)
            import_lines.append('from ai_trading.utils import paths')
            import_lines.append('')
            content = '\n'.join(import_lines + other_lines)
    return content

def replace_async_sleep(content: str) -> str:
    """Replace time.sleep() with await asyncio.sleep() in async functions."""
    lines = content.split('\n')
    modified_lines = []
    in_async_function = False
    for line in lines:
        if re.match('^\\s*(async\\s+)?def\\s+', line):
            in_async_function = 'async' in line
        elif line.strip() and (not line.startswith(' ')) and (not line.startswith('\t')):
            in_async_function = False
        if in_async_function and 'time.sleep(' in line:
            line = line.replace('time.sleep(', 'await asyncio.sleep(')
            if 'await asyncio.sleep(' in line and 'import asyncio' not in '\n'.join(modified_lines):
                import_added = False
                for i, prev_line in enumerate(modified_lines):
                    if prev_line.startswith('import ') or prev_line.startswith('from '):
                        continue
                    else:
                        modified_lines.insert(i, 'import asyncio')
                        import_added = True
                        break
                if not import_added:
                    modified_lines.insert(0, 'import asyncio')
        modified_lines.append(line)
    return '\n'.join(modified_lines)

def process_file(filepath: Path) -> bool:
    """Process a single Python file."""
    try:
        with open(filepath, encoding='utf-8') as f:
            original_content = f.read()
        content = original_content
        content = replace_yaml_load(content)
        content = add_requests_timeout(content)
        content = harden_subprocess(content)
        content = replace_wildcard_imports(content)
        content = replace_data_paths(content)
        content = replace_async_sleep(content)
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except (OSError, PermissionError, KeyError, ValueError, TypeError):
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