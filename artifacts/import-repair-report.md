# Import/Dependency Repair Report

**Date:** <!-- fill in -->
**Commit:** <!-- fill in -->
**Python:** 3.12.3 (Ubuntu 24.04 / glibc 2.39, x86_64)

## Summary
- Total test files collected: <!-- fill in -->
- Import errors: <!-- count -->
- Module-not-found (internal): <!-- list -->
- Module-not-found (external): <!-- list -->
- Skipped (vendor/optional): <!-- list -->

## External packages newly added
- <!-- e.g., scikit-learn==1.5.2, scipy==1.13.1, threadpoolctl==3.5.0, schedule==1.2.2, … -->

## Internal modules restored/stubbed
- `ai_trading.monitoring.performance_monitor.ResourceMonitor`
- `ai_trading.risk.short_selling.validate_short_selling`

## Follow-ups
- RL optional? `RL_AVAILABLE=` <!-- true/false -->
- Any `importorskip` still needed or can be promoted to runtime deps?
- Any tests requiring credentials/network that should be marked `-m integration`?

