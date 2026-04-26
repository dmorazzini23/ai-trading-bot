## Exception Handling Policy

**Goal:** Eliminate broad `except Exception:` in runtime. Use explicit exceptions and structured logs.

### Rules
1. No `except Exception:` in runtime code.
2. Prefer narrow tuples, e.g., `(ValueError, KeyError, json.JSONDecodeError)`.
3. Network/SDK: catch vendor + `requests.exceptions.RequestException`, plus `TimeoutError`.
4. Keep small blocks scoped so the handler covers only the operation that can fail.
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
try:
    resp = client.fetch()
except (requests.exceptions.RequestException, TimeoutError) as exc:
    logger.warning("FETCH_MARKET_DATA_FAILED", extra={"error": str(exc)})
    raise
```
