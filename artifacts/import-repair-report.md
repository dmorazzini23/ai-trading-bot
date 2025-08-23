# Import/Dependency Repair Report

- **OS:** Ubuntu 24.04 (glibc 2.39)
- **Python:** CPython 3.12.3 (x86_64)
- **Wheel tag:** cp312-manylinux_2_39_x86_64
- **Commit:** <!-- fill in -->
- **Date:** <!-- fill in -->

## Summary
- Test files collected: <!-- fill in -->
- Import errors: <!-- fill in -->
- Internal missing modules: <!-- list -->
- External missing packages: <!-- list -->
- Skipped (optional/vendor): <!-- list -->

## External packages newly added
- <!-- e.g., scikit-learn==1.5.2, scipy==1.13.1, ... -->

## Internal modules restored/stubbed
- `ai_trading.position.MarketRegime` (enum + placeholder detector)
- `ai_trading.position.Allocator` / `Allocation` (light protocol)
- `ai_trading.monitoring.performance_monitor.ResourceMonitor`
- `ai_trading.risk.short_selling.validate_short_selling`
