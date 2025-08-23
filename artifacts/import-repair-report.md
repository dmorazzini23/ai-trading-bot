# Import/Dependency Repair Report

**Env**
- Distro: Ubuntu 24.04 (glibc 2.39)
- Python: CPython 3.12.3
- Wheel tag: `cp312-manylinux_2_39_x86_64`

## Summary
- Remaining import errors (unique): <!--COUNT-->
- Modules:
<!--IMPORT_ERRORS-->

## Notes
- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map.
- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only).
