    def _coverage_threshold_scales(feed: str | None) -> tuple[float, float]:
        normalized_feed = _normalize_feed_name(feed)
        if not normalized_feed:
            normalized_feed = _normalize_feed_name(
                configured_feed or os.getenv("ALPACA_DATA_FEED")
            )
        if normalized_feed == "iex":
            return 0.75, 0.75
        return 1.0, 1.0

    def _window_minutes(start: datetime, end: datetime) -> int:
        try:
            return max(0, int((end - start).total_seconds() // 60))
        except COMMON_EXC:
            return 0

        window_minutes: int,
        threshold_scale, intraday_scale = _coverage_threshold_scales(active_feed)
        threshold = 0
        if expected > 0:
            threshold = max(1, int(expected * 0.5 * threshold_scale))
        cutoff = intraday_requirement
        if intraday_requirement > 0:
            cutoff = max(1, int(intraday_requirement * intraday_scale))
        normalized_feed = _normalize_feed_name(active_feed)
        insufficient_local = expected >= cutoff and actual < cutoff
        if (
            normalized_feed == "sip"
            and window_minutes >= cutoff
            and actual < cutoff
        ):
            insufficient_local = True
        window_minutes=_window_minutes(start_dt, end_dt),
                    window_minutes=_window_minutes(fallback_start_dt, end_dt),
                    window_minutes=_window_minutes(fallback_start_dt, end_dt),
        window_minutes=_window_minutes(start_dt, end_dt),
                window_minutes=_window_minutes(start_dt, end_dt),
