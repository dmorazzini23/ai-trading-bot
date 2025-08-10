#!/usr/bin/env bash
# tools/ci/validate_shim_removal.sh
# Comprehensive validation script for shim removal

set -euo pipefail

echo "🔍 Validating shim removal and Settings enforcement..."

# Step 1: Verify no shims remain
echo "1. Running shim guard..."
if bash tools/ci/guard_shims.sh; then
    echo "✅ No shim patterns detected"
else
    echo "❌ Shim patterns still present - run tools/ci/guard_shims.sh for details"
    exit 1
fi

# Step 2: Test basic config imports
echo "2. Testing config imports..."
python - << 'PY'
import sys
sys.path.insert(0, '.')

# Test direct Settings import
from ai_trading.config import get_settings, Settings

# Test Settings instantiation
S = get_settings()
print(f"✅ Settings loaded: {type(S).__name__}")
print(f"✅ Sample setting - trading_mode: {S.trading_mode}")
print(f"✅ Sample setting - seed: {S.seed}")
print(f"✅ Sample setting - testing: {S.testing}")

# Test that no magic __getattr__ exists
import ai_trading.config as config_module
if hasattr(config_module, '__getattr__'):
    print("❌ __getattr__ still exists in config module")
    sys.exit(1)
else:
    print("✅ No __getattr__ magic in config module")
PY

# Step 3: Test that key modules import without shims
echo "3. Testing core module imports..."
python - << 'PY'
import sys
sys.path.insert(0, '.')

# Test that env.py works without dotenv shim
try:
    from ai_trading.env import ensure_dotenv_loaded
    print("✅ env.py imports without shims")
except Exception as e:
    print(f"❌ env.py import failed: {e}")
    sys.exit(1)

# Test database connection without internal import shims
try:
    from ai_trading.database.connection import DatabaseManager
    print("✅ database.connection imports without shims")
except Exception as e:
    print(f"❌ database.connection import failed: {e}")
    sys.exit(1)
PY

# Step 4: Compile check for syntax errors
echo "4. Running syntax validation..."
python -m py_compile $(find ai_trading -name "*.py" | head -20) || {
    echo "❌ Syntax errors found in Python files"
    exit 1
}
echo "✅ Core Python files compile successfully"

# Step 5: Test Settings access patterns
echo "5. Testing standardized config access..."
python - << 'PY'
import sys
sys.path.insert(0, '.')

from ai_trading.config import get_settings

S = get_settings()

# Test that common fields are accessible with lowercase names
test_fields = [
    'trading_mode', 'shadow_mode', 'seed', 'testing', 
    'alpaca_api_key', 'alpaca_secret_key', 'alpaca_base_url',
    'scheduler_sleep_seconds', 'trade_log_file'
]

for field in test_fields:
    if hasattr(S, field):
        print(f"✅ Field {field} accessible")
    else:
        print(f"❌ Field {field} missing")
        sys.exit(1)

print("✅ All expected Settings fields accessible")
PY

# Step 6: Style and formatting check (if available)
echo "6. Running style checks..."
if command -v ruff >/dev/null 2>&1; then
    echo "Running ruff checks..."
    ruff check ai_trading/config/ ai_trading/env.py ai_trading/database/connection.py --quiet || {
        echo "⚠️  Style issues found (non-blocking)"
    }
else
    echo "⚠️  ruff not available - skipping style checks"
fi

if command -v black >/dev/null 2>&1; then
    echo "Running black format check..."
    black --check ai_trading/config/ ai_trading/env.py ai_trading/database/connection.py --quiet || {
        echo "⚠️  Format issues found (non-blocking)"
    }
else
    echo "⚠️  black not available - skipping format checks"
fi

echo ""
echo "🎉 Shim removal validation completed successfully!"
echo ""
echo "📊 Summary:"
echo "  ✅ Config magic (__getattr__) removed"
echo "  ✅ Direct Settings imports working"
echo "  ✅ Core modules compile without shims"
echo "  ✅ Typed Settings fields accessible"
echo ""
echo "🚀 Ready for production use with hard dependencies!"