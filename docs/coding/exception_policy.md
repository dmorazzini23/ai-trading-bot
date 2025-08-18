## Exception Handling Policy

**Goal:** Eliminate broad `except Exception:` in runtime. Use explicit exceptions and structured logs.

### Rules
1. No `except Exception:` in runtime code.
2. Prefer narrow tuples, e.g., `(ValueError, KeyError, json.JSONDecodeError)`.
3. Network/SDK: catch vendor + `requests.exceptions.RequestException`, plus `TimeoutError`.
4. Use `guards.catch((...), label="...")` for small blocks; never pass `Exception`.
5. Log context (`label`) and re-raise when the caller can handle it.

### Example (before)
```py
try:
    resp = client.fetch()
except Exception as e:
    logger.warning("fetch failed: %s", e)
```

### Example (after)
```py
from ai_trading.utils import guards
with guards.catch((requests.exceptions.RequestException, TimeoutError), label="fetch-market-data"):
    resp = client.fetch()
```

