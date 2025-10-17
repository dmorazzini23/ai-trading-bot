from ai_trading.data import provider_monitor as monitor_mod


def test_provider_logs_use_canonical_labels(caplog):
    caplog.set_level("INFO", logger=monitor_mod.logger.name)

    monitor_mod.record_stay(provider="alpaca_iex", reason="stay", cooldown=10)
    monitor_mod.record_stay(provider="yfinance", reason="stay", cooldown=10)

    stay_messages = [record.message for record in caplog.records if "DATA_PROVIDER_STAY" in record.message]
    assert any("provider=alpaca-iex" in message for message in stay_messages)
    assert any("provider=yahoo" in message for message in stay_messages)

    caplog.clear()
    caplog.set_level("INFO", logger=monitor_mod.logger.name)

    monitor = monitor_mod.ProviderMonitor(cooldown=0, threshold=1)
    monitor.record_switchover("alpaca_alpaca", "alpaca_yfinance")

    switchover_records = [
        record for record in caplog.records if record.message.startswith("DATA_PROVIDER_SWITCHOVER")
    ]
    assert switchover_records, "expected DATA_PROVIDER_SWITCHOVER log"
    extras = switchover_records[0].__dict__
    assert extras.get("from_provider") == "alpaca"
    assert extras.get("to_provider") == "yahoo"
