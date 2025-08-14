import re, sys, pathlib

DENY_BARE = {
    "order_health_monitor",
    "trade_execution",      # bare
    "strategy_allocator",   # bare
    "ml_model",             # bare
    "risk_engine",          # bare
    "performance_monitor",  # bare
    "validate_env",         # bare
    "_require_env_vars",
    "check_data_freshness", # bare
    "system_health_checker",# bare
    "portfolio_optimizer",  # bare
    "transaction_cost_calculator",
    "process_manager",      # bare
}

DENY_PACKAGE = {
    "ai_trading.trade_execution",
    "ai_trading.monitoring.performance_monitor",
    "ai_trading.validation.require_env",
    "ai_trading.validation.check_data_freshness",
    "ai_trading.utils.process_manager",
    "ai_trading.thirdparty.lightgbm_compat",
}

ROOT = pathlib.Path("tests")
pat = re.compile(r"^\s*(from|import)\s+(" + "|".join(map(re.escape, DENY_BARE)) + r")\b")

bad = []
for p in ROOT.rglob("*.py"):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if pat.search(line):
            bad.append(f"{p}:{i}:{line.strip()}")

if bad:
    print("[contract] Legacy imports found in tests:")
    print("\n".join(bad))
    sys.exit(1)

pat_pkg = re.compile(r"^\s*(from|import)\s+(" + "|".join(map(re.escape, DENY_PACKAGE)) + r")\b")

bad_pkg = []
for p in pathlib.Path("ai_trading").rglob("*.py"):
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if pat_pkg.search(line):
            bad_pkg.append(f"{p}:{i}:{line.strip()}")

if bad_pkg:
    print("[contract] Forbidden package imports found:\n" + "\n".join(bad_pkg))
    sys.exit(1)

