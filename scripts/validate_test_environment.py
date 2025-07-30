#!/usr/bin/env python3
"""
Validate that all required test dependencies are available.
This script helps identify missing dependencies before running tests.
"""

import sys
import importlib
from typing import List, Tuple

# Required packages for testing
REQUIRED_PACKAGES = [
    ('pytest', 'pytest'),
    ('pytest_cov', 'pytest-cov'), 
    ('pytest_xdist', 'pytest-xdist'),
    ('pytest_benchmark', 'pytest-benchmark'),
    ('pytest_asyncio', 'pytest-asyncio'),
    ('hypothesis', 'hypothesis'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('sklearn', 'scikit-learn'),
    ('joblib', 'joblib'),
    ('pyarrow', 'pyarrow'),
    ('stable_baselines3', 'stable-baselines3'),
    ('gymnasium', 'gymnasium'),
    ('flake8', 'flake8'),
    ('mypy', 'mypy'),
]

def check_package(import_name: str, package_name: str) -> Tuple[bool, str]:
    """Check if a package can be imported."""
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    """Main validation function."""
    print("🔍 Validating test environment dependencies...")
    print("=" * 50)
    
    missing_packages = []
    available_packages = []
    
    for import_name, package_name in REQUIRED_PACKAGES:
        success, info = check_package(import_name, package_name)
        
        if success:
            available_packages.append((package_name, info))
            print(f"✅ {package_name}: {info}")
        else:
            missing_packages.append((package_name, info))
            print(f"❌ {package_name}: {info}")
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   Available: {len(available_packages)}")
    print(f"   Missing:   {len(missing_packages)}")
    
    if missing_packages:
        print(f"\n🚨 Missing packages:")
        for package_name, error in missing_packages:
            print(f"   - {package_name}")
        
        print(f"\n💡 To install missing packages:")
        print(f"   pip install -r requirements-dev.txt")
        
        return 1
    else:
        print(f"\n🎉 All test dependencies are available!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)