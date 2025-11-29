        window_limit = min_interval if min_interval > 0 else ttl_window

        def _short_circuit_memo_entry(key: tuple[str, ...]) -> bool:
            nonlocal memo_ready, memo_hit, cached_df, cached_reason, refresh_stamp, refresh_df, refresh_source
            entry_ts, entry_payload, normalized_pair = _extract_memo_payload(
                _memo_get_entry(key)
            )
            if entry_payload is None:
                return False
            if entry_ts is not None and entry_ts <= 0.0:
                return False
            age = None if entry_ts is None else now_monotonic - entry_ts
            is_fresh = age is None or age <= memo_ttl_limit or age <= window_limit
            if not is_fresh:
                return False
            normalized_now = normalized_pair or (now_monotonic, entry_payload)
            memo_ready = True
            memo_hit = True
            cached_df = entry_payload
            cached_reason = "memo"
            refresh_stamp = normalized_now[0]
            refresh_df = entry_payload
            refresh_source = "memo"
            with cache_lock:
                _memo_set_entry(canonical_memo_key, normalized_now)
                _memo_set_entry(memo_key, normalized_now)
                _memo_set_entry(legacy_memo_key, normalized_now)
            return True

        if _short_circuit_memo_entry(canonical_memo_key):
            return _finalize_cached_return()
        if _short_circuit_memo_entry(legacy_memo_key):
            return _finalize_cached_return()

