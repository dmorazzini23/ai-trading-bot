#!/bin/bash
# PDT Fix Deployment Script
# This script installs dependencies and deploys the PDT integration fix

set -e  # Exit on error

echo "================================================================================"
echo "PDT FIX DEPLOYMENT SCRIPT"
echo "================================================================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "1. Checking current branch..."
CURRENT_BRANCH=$(git branch --show-current)
echo "   Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "fix/pdt-complete-solution" ]; then
    echo "   ⚠️  Not on fix/pdt-complete-solution branch"
    echo "   Switching to fix/pdt-complete-solution..."
    git checkout fix/pdt-complete-solution
fi

echo ""
echo "2. Pulling latest changes..."
git pull origin fix/pdt-complete-solution || echo "   (Branch may not exist on remote yet)"

echo ""
echo "3. Installing missing dependencies..."
if [ -f "requirements_pdt_fix.txt" ]; then
    pip3 install -r requirements_pdt_fix.txt --quiet
    echo "   ✅ Dependencies installed"
else
    echo "   ⚠️  requirements_pdt_fix.txt not found, skipping..."
fi

echo ""
echo "4. Verifying PDT integration..."
if [ -f "verify_pdt_integration.py" ]; then
    python3 verify_pdt_integration.py
    VERIFY_EXIT=$?
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo "   ✅ Verification passed"
    else
        echo "   ❌ Verification failed (exit code: $VERIFY_EXIT)"
        echo "   Please review the errors above before proceeding"
        exit 1
    fi
else
    echo "   ⚠️  verify_pdt_integration.py not found, skipping verification..."
fi

echo ""
echo "5. Checking if service exists..."
if systemctl list-units --full -all | grep -q "ai-trading.service"; then
    echo "   Service found: ai-trading.service"
    echo ""
    echo "6. Restarting trading bot service..."
    sudo systemctl restart ai-trading.service
    echo "   ✅ Service restarted"
    
    echo ""
    echo "7. Checking service status..."
    sudo systemctl status ai-trading.service --no-pager -l | head -15
    
    echo ""
    echo "8. Monitoring logs for PDT integration..."
    echo "   Looking for PDT_STATUS_CHECK in recent logs..."
    sleep 3
    
    if sudo journalctl -u ai-trading.service -n 100 --no-pager | grep -q "PDT_STATUS_CHECK"; then
        echo "   ✅ PDT integration is ACTIVE!"
        echo ""
        echo "   Recent PDT-related log messages:"
        sudo journalctl -u ai-trading.service -n 100 --no-pager | grep -E "PDT_STATUS_CHECK|PDT_LIMIT_EXCEEDED|SWING_MODE" | tail -5
    else
        echo "   ⚠️  PDT_STATUS_CHECK not found in recent logs"
        echo "   The integration may not be active yet. Check logs manually:"
        echo "   sudo journalctl -u ai-trading.service -f"
    fi
else
    echo "   ⚠️  ai-trading.service not found"
    echo "   You'll need to restart the bot manually"
fi

echo ""
echo "================================================================================"
echo "DEPLOYMENT COMPLETE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Monitor logs: sudo journalctl -u ai-trading.service -f"
echo "  2. Look for: PDT_STATUS_CHECK, PDT_LIMIT_EXCEEDED_SWING_MODE_ACTIVATED"
echo "  3. Verify orders are executing (check broker positions)"
echo ""
echo "If you see PDT_STATUS_CHECK in logs, the fix is working! ✅"
echo ""

