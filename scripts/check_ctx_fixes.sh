#!/bin/bash
# Static checks for ctx usage fixes

echo "Running static checks for bare ctx usage fixes..."

cd "$(dirname "$0")/.."

# Check 1: No check_pdt_rule calls with ctx parameter
echo "Checking for check_pdt_rule(ctx) calls..."
if grep -r "check_pdt_rule\s*(\s*ctx\s*)" ai_trading/; then
    echo "✗ Found check_pdt_rule(ctx) calls"
    exit 1
else
    echo "✓ No check_pdt_rule(ctx) calls found"
fi

# Check 2: run_all_trades_worker signature should not have 'model' as second parameter
echo "Checking run_all_trades_worker signature..."
if grep -n "def run_all_trades_worker.*model" ai_trading/core/bot_engine.py; then
    echo "✗ run_all_trades_worker still has 'model' parameter"
    exit 1
else
    echo "✓ run_all_trades_worker signature updated"
fi

# Check 3: runner.py should not pass None to run_all_trades_worker
echo "Checking runner.py for None parameter..."
if grep -n "run_all_trades_worker.*None" ai_trading/runner.py; then
    echo "✗ runner.py still passes None to run_all_trades_worker"
    exit 1
else
    echo "✓ runner.py updated to pass runtime context"
fi

# Check 4: Key functions should have runtime parameter instead of ctx
echo "Checking helper function signatures..."
functions=("check_pdt_rule" "cancel_all_open_orders" "audit_positions" "_log_health_diagnostics" "_prepare_run")
for func in "${functions[@]}"; do
    if grep -n "def $func.*ctx:" ai_trading/core/bot_engine.py; then
        echo "✗ Function $func still has ctx parameter type annotation"
        exit 1
    fi
done
echo "✓ Helper function signatures updated"

echo ""
echo "All static checks passed!"
echo "OK"