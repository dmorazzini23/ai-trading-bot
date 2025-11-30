    try:
        _ensure_alpaca_classes()
    except COMMON_EXC:
        if not (getattr(CFG, "testing", False) or pytest_running):
            raise
        try:
            _ensure_alpaca_classes()
        except COMMON_EXC:
            if pytest_running:
                return types.SimpleNamespace(**args)
            raise
        if any(
            not callable(cls)
            for cls in (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
            )
        ):
            return types.SimpleNamespace(**args)
        except (*COMMON_EXC, AttributeError):
