# Fix Summary: Message-Shortening Ellipsis and Risk Exposure Task

## Problem Statement
- JSON logs showed Unicode ellipsis (`\u2026`) in `msg` fields due to `ensure_ascii=True`  
- Periodic "risk engine exposure update" job referenced undefined module-scope `ctx`, causing minute-interval warnings

## Solutions Implemented

### 1. Fixed JSON Logging Unicode Handling (`ai_trading/logging.py`)

**Changes:**
- Line 96: `json.dumps(payload, default=self._json_default, ensure_ascii=False)`
- Line 434: `json.dumps(log_entry, default=str, separators=(",", ":"), ensure_ascii=False)`

**Result:**
- Unicode characters like `—` and `…` now display properly instead of `\u2014` and `\u2026`
- Log messages preserve full Unicode fidelity for international characters

### 2. Added Risk Exposure Update Task (`ai_trading/core/bot_engine.py`)

**New Functions Added:**
```python
def _get_runtime_context_or_none():
    """Safely acquire runtime context from LazyBotContext singleton."""
    
def _update_risk_engine_exposure():
    """Periodic task that updates risk engine without module-scope ctx dependency."""
```

**Scheduled Task Added:**
```python
schedule.every(1).minutes.do(
    lambda: Thread(target=_update_risk_engine_exposure, daemon=True).start()
)
```

**Features:**
- Uses runtime context accessor instead of module-scope `ctx`
- Logs warnings only once to avoid spam
- Returns quietly if context unavailable
- Handles missing risk_engine gracefully

## Validation

### Tests Created
- `tests/test_ellipsis_fix.py` - Comprehensive unit tests for both fixes
- `validate_fixes.py` - Standalone validation script

### Validation Results
✅ All Python files compile successfully (`compileall` passes)  
✅ Unicode characters preserved: `—` instead of `\u2014`, `…` instead of `\u2026`  
✅ Risk exposure functions properly defined and scheduled  
✅ No more "name 'ctx' is not defined" warnings expected  
✅ JSON formatter maintains existing functionality while fixing Unicode  

## Before/After Comparison

### JSON Logging
**Before:** `{"msg": "MARKET WATCH \u2014 Status \u2026 Complete"}`  
**After:** `{"msg": "MARKET WATCH — Status … Complete"}`

### Risk Exposure Task
**Before:** `NameError: name 'ctx' is not defined` (every minute)  
**After:** Silent operation with proper runtime context handling

## Production Impact
- **Low Risk:** Changes are minimal and focused
- **Backward Compatible:** JSON structure unchanged, only Unicode encoding improved
- **Performance:** No performance impact, same functionality with better Unicode handling
- **Monitoring:** Reduced log noise from ctx errors

## Rollback Plan
If needed, revert by:
1. Change `ensure_ascii=False` back to `ensure_ascii=True` (or remove parameter)
2. Remove `_update_risk_engine_exposure` and `_get_runtime_context_or_none` functions
3. Remove the scheduled task for risk exposure updates