# Implementation Strategy for Remaining Shim Removal

## Current Status
After our changes, the shim guard shows:
- ✅ **Config magic eliminated**: No `__getattr__` functions remain
- ❌ **Import guards**: 117+ files still have try/except ImportError patterns  
- ❌ **Mock classes**: 59+ Mock* classes still in runtime code
- ✅ **No eval/exec issues**: Only PyTorch `.eval()` method calls detected

## Strategy for Import Guard Removal

### Phase 1: Safe Internal Imports (DONE)
- ✅ Fixed ai_trading/env.py - removed dotenv import guard
- ✅ Fixed ai_trading/database/connection.py - removed internal import guard

### Phase 2: Essential Dependencies (REQUIRES ENVIRONMENT SETUP)
Files with import guards for core dependencies that should be hard requirements:

1. **High Priority - Core Libraries**:
   - `numpy` - Used in 20+ files, should be hard dependency
   - `pandas` - Used in 15+ files, should be hard dependency  
   - `scikit-learn` - Used in ML modules, should be hard dependency

2. **Medium Priority - Domain Libraries**:
   - `alpaca-trade-api` - Core trading functionality (use a single SDK)
   - `yfinance` - Market data (but could be optional with feature flags)
   - `flask` - Web interface (could be optional)

3. **Low Priority - Optional Features**:
   - Advanced ML libraries (`lightgbm`, `xgboost`)
   - Specialized analysis tools (`ta`, `beautifulsoup4`)

### Phase 3: Mock Class Relocation Strategy

#### Files with Heavy Mock Usage (Priority Order):
1. **ai_trading/imports.py** - Central import management (59 Mock classes)
2. **ai_trading/core/bot_engine.py** - Core engine (20+ Mock classes)
3. **ai_trading/strategies/imports.py** - Strategy dependencies (8 Mock classes)
4. **ai_trading/position/*.py** - Position management (12 files, 2 Mock classes each)

#### Relocation Approach:
```python
# BEFORE (in runtime file):
try:
    import numpy as np
except ImportError:
    class MockNumpy:
        # mock implementation
    np = MockNumpy()

# AFTER (hard dependency):
import numpy as np

# Mock moved to tests/support/mocks_runtime.py for test usage only
```

## Implementation Commands

### Run when environment has dependencies:
```bash
# Remove import guards automatically
python tools/codemods/remove_import_guards.py

# Manual verification and fixes
./tools/ci/guard_shims.sh

# Move Mock classes to tests
# (Manual process - need to check each usage)

# Validate final result
./tools/ci/validate_shim_removal.sh
```

### Validation Checklist:
- [ ] No try/except ImportError blocks in ai_trading/
- [ ] No Mock* classes in ai_trading/ (moved to tests/support/)
- [ ] All Settings access uses get_settings() with lowercase fields
- [ ] Package imports cleanly with hard dependencies
- [ ] Core functionality tests pass

## Production Guidance

Runtime code must not use `optional_import(...)`. Instead, rely on `try`/`except ImportError` or `importlib.util.find_spec` and gate heavy imports inside function scope when rarely used.

## Risk Mitigation

### Fallback Strategy:
If removing import guards breaks functionality:
1. Identify specific missing dependencies
2. Add them to pyproject.toml [dependencies]
3. Consider feature flags for truly optional components

### Testing Strategy:
1. Run validation script after each change
2. Test with minimal dependencies first
3. Gradually remove guards from least critical to most critical files
4. Maintain CI guard to prevent regression

## Final Validation

The package should:
- ✅ Import without any try/except ImportError blocks
- ✅ Use typed Settings for all configuration
- ✅ Have Mock classes only in tests/
- ✅ Fail fast if hard dependencies are missing
- ✅ Pass the shim guard script