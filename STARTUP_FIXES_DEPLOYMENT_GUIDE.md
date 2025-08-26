# Startup Fixes Implementation - Deployment Guide

## Overview

This implementation addresses import-time crashes and ensures predictable startup under systemd by implementing:

1. **Deferred credential validation** - No sys.exit during import
2. **Early .env loading** - Before heavy module imports and Settings construction  
3. **Dual credential schema support** - Both ALPACA_* and APCA_* variable names
4. **UTC timestamp fixes** - Single "Z" suffix instead of double "ZZ"
5. **Lazy imports** - Heavy modules loaded only when needed
6. **Comprehensive testing** - Prevents regression

## Validation Steps

### 1. Pre-deployment Validation

Run the comprehensive validation script:

```bash
cd /path/to/ai-trading-bot
python validate_startup_fixes.py
```

Expected output:
```
🎉 ALL TESTS PASSED!
✓ Service no longer crashes at import
✓ Bot starts with either ALPACA_* or APCA_* credentials
✓ Credentials are handled securely with redacted logging
✓ UTC timestamps have single trailing Z (no 'ZZ')
✓ Lazy imports prevent import-time side effects
✓ Backward compatibility maintained

🚀 Ready for systemd deployment!
```

### 2. Systemd Deployment Validation

After deployment, validate with:

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Restart the service
sudo systemctl restart ai-trading.service

# Check service status (should not show import failures)
sudo systemctl status ai-trading.service

# Monitor logs for startup behavior
sudo journalctl -u ai-trading.service -n 200 -f
```

### 3. Environment Variable Validation

Check that credentials are loaded correctly:

```bash
# Get the service PID
PID=$(systemctl show -p MainPID --value ai-trading.service)

# Check environment variables (should show either ALPACA_* or APCA_* variables)
sudo tr '\0' '\n' </proc/$PID/environ | grep -E 'APCA|ALPACA|BASE_URL'
```

### 4. Test Environment Integration

Run the focused tests:

```bash
# Activate virtual environment
source venv/bin/activate

# Run specific test categories
pytest -q -k "env_order or dual_schema or utc_timefmt"
```

## Key Implementation Changes

### 1. Removed Import-Time Validation

**Before:**
```python
# ai_trading/core/bot_engine.py (BROKEN)
if not (API_KEY and API_SECRET) and not config.is_shadow_mode():
    logger.critical("Alpaca credentials missing – aborting startup")
    sys.exit(1)  # ❌ Crashes during import
```

**After:**
```python
# ai_trading/core/bot_engine.py (FIXED)
def _initialize_alpaca_clients():
    """Initialize clients with runtime credential validation."""
    try:
        api_key, secret_key, base_url = _ensure_alpaca_env_or_raise()
    except RuntimeError as e:
        if config.is_shadow_mode():
            logger.warning("Running in SHADOW_MODE with missing credentials: %s", e)
            return
        else:
            logger.critical("Alpaca credentials missing – cannot initialize clients")
            raise e  # ✅ Runtime validation only
```

### 2. Dual Credential Schema Support

**New function in config/management.py:**
```python
def _resolve_alpaca_env() -> tuple[str | None, str | None, str | None]:
    """Resolve credentials supporting both ALPACA_* and APCA_* schemes."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") 
    base_url = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL")
    # ... (ALPACA_* takes precedence)
```

### 3. Early .env Loading

**In main.py:**
```python
# AI-AGENT-REF: Load .env BEFORE importing any heavy modules or Settings
from dotenv import load_dotenv
load_dotenv(override=True)

# AI-AGENT-REF: Import Settings AFTER .env is loaded
from ai_trading.config import Settings
```

### 4. Lazy Import Mechanism

**In runner.py:**
```python
def _load_engine():
    """Lazy loader for bot engine components."""
    global _bot_engine, _bot_state_class
    
    if _bot_engine is None or _bot_state_class is None:
        # Import only when needed
        from ai_trading.core.bot_engine import run_all_trades_worker, BotState
        _bot_engine = run_all_trades_worker
        _bot_state_class = BotState
```

### 5. UTC Timestamp Fixes

**New utility in utils/timefmt.py:**
```python
def utc_now_iso() -> str:
    """Generate UTC timestamp with single 'Z' suffix."""
    now = datetime.now(timezone.utc)
    return now.isoformat().replace('+00:00', 'Z')  # ✅ Single Z
```

## Environment Variable Support

Both naming conventions are now supported:

| ALPACA_* (Preferred) | APCA_* (Alternative) | Purpose |
|---------------------|---------------------|---------|
| `ALPACA_API_KEY` | `APCA_API_KEY_ID` | API key |
| `ALPACA_SECRET_KEY` | `APCA_API_SECRET_KEY` | Secret key |
| `ALPACA_BASE_URL` | `APCA_API_BASE_URL` | Base URL |

**Precedence:** ALPACA_* variables take precedence if both are present.

## Error Handling

### Import-Time Behavior (Fixed)
- ✅ No sys.exit() calls during import
- ✅ Missing credentials don't prevent module loading
- ✅ Heavy imports are deferred until runtime

### Runtime Behavior  
- ✅ Credentials validated when clients are initialized
- ✅ Clear error messages for missing credentials
- ✅ Graceful fallback in SHADOW_MODE

## Backward Compatibility

All existing functionality is preserved:
- ✅ Existing ALPACA_* variables continue to work
- ✅ No breaking API changes
- ✅ Same logging behavior (with improved redaction)
- ✅ All existing configuration options supported

## Troubleshooting

### Service Fails to Start

1. Check systemd logs:
   ```bash
   sudo journalctl -u ai-trading.service -n 50
   ```

2. Verify credentials are set:
   ```bash
   # Should show masked values
   sudo systemctl show-environment | grep -E 'ALPACA|APCA'
   ```

3. Test import manually:
   ```bash
   cd /path/to/ai-trading-bot
   python -c "from ai_trading import runner; print('✓ Import successful')"
   ```

### Credential Issues

1. Check both naming schemes:
   ```bash
   echo "ALPACA_API_KEY: ${ALPACA_API_KEY:0:8}***"
   echo "APCA_API_KEY_ID: ${APCA_API_KEY_ID:0:8}***"
   ```

2. Verify .env file is loaded:
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(override=True); import os; print('ALPACA_API_KEY set:', bool(os.getenv('ALPACA_API_KEY')))"
   ```

### UTC Timestamp Issues

1. Check for double Z in logs:
   ```bash
   sudo journalctl -u ai-trading.service | grep -o '[0-9T:-]*ZZ'
   ```

2. Test timestamp utility:
   ```bash
   python -c "from ai_trading.utils.timefmt import utc_now_iso; print(utc_now_iso())"
   ```

## Success Criteria

✅ **All validation tests pass**  
✅ **Service starts without import-time crashes**  
✅ **Both credential schemas work**  
✅ **UTC timestamps have single Z**  
✅ **Lazy imports prevent side effects**  
✅ **Backward compatibility maintained**

The implementation is ready for production systemd deployment!