from __future__ import annotations

import sys

EXPECTED_MAJOR = 3
EXPECTED_MINOR = 12


def check_version(version_info: tuple[int, ...] = sys.version_info[:3]) -> tuple[bool, str]:
    major, minor = version_info[:2]
    expected = f"{EXPECTED_MAJOR}.{EXPECTED_MINOR}"
    observed = ".".join(str(part) for part in version_info[:3])
    if (major, minor) == (EXPECTED_MAJOR, EXPECTED_MINOR):
        return True, f"Python {observed} matches required {expected}"
    return False, f"Python {expected} is required; found {observed}"


def main() -> int:
    ok, message = check_version()
    stream = sys.stdout if ok else sys.stderr
    print(message, file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
