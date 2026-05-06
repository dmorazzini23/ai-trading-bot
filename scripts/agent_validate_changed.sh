#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

MODE="targeted"
MARKET_HOURS=0
DOCS_ONLY_OVERRIDE=0
FORCE_BROAD=0
SKIP_RUNTIME_SMOKE=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/agent_validate_changed.sh [--targeted|--full] [options]

Options:
  --targeted             Validate changed files only (default).
  --full                 Run the full repository validation suite.
  --market-hours         Refuse broad validation unless --force-broad is also set.
  --docs-only            Treat this as documentation-only validation.
  --force-broad          Allow --full with --market-hours.
  --skip-runtime-smoke   Skip live /healthz and alert snapshot smoke checks.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targeted)
      MODE="targeted"
      ;;
    --full)
      MODE="full"
      ;;
    --market-hours)
      MARKET_HOURS=1
      ;;
    --docs-only)
      DOCS_ONLY_OVERRIDE=1
      ;;
    --force-broad)
      FORCE_BROAD=1
      ;;
    --skip-runtime-smoke)
      SKIP_RUNTIME_SMOKE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$MODE" == "full" && "$MARKET_HOURS" == "1" && "$FORCE_BROAD" != "1" ]]; then
  echo "refusing full validation during market hours; pass --force-broad only with explicit approval" >&2
  exit 2
fi

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run() {
  echo "+ $*"
  "$@"
}

collect_changed_files() {
  local files=()
  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    mapfile -t files < <(git diff --name-only --diff-filter=ACMRTUXB HEAD --)
  else
    mapfile -t files < <(git diff --name-only --diff-filter=ACMRTUXB --)
  fi
  local untracked=()
  mapfile -t untracked < <(git ls-files --others --exclude-standard)
  printf '%s\n' "${files[@]}" "${untracked[@]}" | awk 'NF' | sort -u
}

is_doc_file() {
  local file="$1"
  case "$file" in
    *.md|*.rst|*.txt|docs/*|README*|ARCHITECTURE.md|API_DOCUMENTATION.md|DEPLOYING.md|AGENTS.md)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

path_exists() {
  [[ -e "$1" ]]
}

add_existing_test_target() {
  local target="$1"
  if [[ -e "$target" ]]; then
    TEST_TARGETS["$target"]=1
  fi
}

add_related_tests_for_file() {
  local file="$1"
  case "$file" in
    tests/*.py|tests/*/*.py|tests/*/*/*.py)
      add_existing_test_target "$file"
      ;;
    tools/mcp_slack_alerts_server.py)
      add_existing_test_target "tests/tools/test_mcp_slack_alerts_server.py"
      ;;
    tools/*.py)
      add_existing_test_target "tests/tools"
      ;;
    scripts/*.sh|scripts/*.py)
      add_existing_test_target "tests/scripts"
      ;;
    ai_trading/app.py|ai_trading/health*.py|ai_trading/core/runtime_services.py)
      add_existing_test_target "tests/health"
      add_existing_test_target "tests/app"
      add_existing_test_target "tests/core/test_runtime_services.py"
      ;;
    ai_trading/risk/*)
      add_existing_test_target "tests/risk"
      ;;
    ai_trading/execution/*)
      add_existing_test_target "tests/execution"
      add_existing_test_target "tests/core/test_execution_engine_runtime.py"
      ;;
    ai_trading/data/*)
      add_existing_test_target "tests/data"
      ;;
    ai_trading/features/*)
      add_existing_test_target "tests/features"
      ;;
    ai_trading/training/*)
      add_existing_test_target "tests/training"
      ;;
    ai_trading/market/*)
      add_existing_test_target "tests/market"
      ;;
    ai_trading/config/*)
      add_existing_test_target "tests/config"
      ;;
    ai_trading/core/*)
      add_existing_test_target "tests/core"
      ;;
    ai_trading/portfolio/*)
      add_existing_test_target "tests/portfolio"
      ;;
    ai_trading/signals/*)
      add_existing_test_target "tests/signals"
      ;;
    ai_trading/oms/*)
      add_existing_test_target "tests/oms"
      ;;
    ai_trading/governance/*)
      add_existing_test_target "tests/governance"
      ;;
    ai_trading/replay/*)
      add_existing_test_target "tests/runtime"
      add_existing_test_target "tests/governance"
      ;;
    ai_trading/*)
      local module
      module="$(cut -d/ -f2 <<<"$file")"
      add_existing_test_target "tests/$module"
      ;;
  esac
}

run_full_validation() {
  run ./venv/bin/pytest -q
  run ./venv/bin/ruff check
  run ./venv/bin/mypy
  run bash scripts/typecheck_strict.sh
  run /bin/bash -lc "./venv/bin/python -m py_compile \$(git ls-files '*.py')"
}

if [[ "$MODE" == "full" ]]; then
  run_full_validation
  echo "agent_validate_changed: full validation passed"
  exit 0
fi

mapfile -t CHANGED_FILES < <(collect_changed_files)

if [[ "${#CHANGED_FILES[@]}" -eq 0 ]]; then
  echo "agent_validate_changed: no changed files detected"
  exit 0
fi

declare -a PY_FILES=()
declare -a PY_EXISTING_FILES=()
declare -a SHELL_FILES=()
declare -a SYSTEMD_FILES=()
declare -a DEPENDENCY_FILES=()
declare -a RUNTIME_SMOKE_FILES=()
declare -A TEST_TARGETS=()
HAS_CODE_CHANGE=0
ALL_DOCS=1

for file in "${CHANGED_FILES[@]}"; do
  if ! is_doc_file "$file"; then
    ALL_DOCS=0
  fi
  case "$file" in
    *.py)
      PY_FILES+=("$file")
      if path_exists "$file"; then
        PY_EXISTING_FILES+=("$file")
      fi
      HAS_CODE_CHANGE=1
      add_related_tests_for_file "$file"
      ;;
    *.sh)
      SHELL_FILES+=("$file")
      HAS_CODE_CHANGE=1
      add_related_tests_for_file "$file"
      ;;
    packaging/systemd/*.service|packaging/systemd/*.timer)
      SYSTEMD_FILES+=("$file")
      HAS_CODE_CHANGE=1
      ;;
    pyproject.toml|requirements*.txt|constraints*.txt|setup.cfg|setup.py)
      DEPENDENCY_FILES+=("$file")
      HAS_CODE_CHANGE=1
      ;;
  esac
  case "$file" in
    ai_trading/*|tools/mcp_slack_alerts_server.py|scripts/connector_incident_dispatch.py|packaging/systemd/*)
      RUNTIME_SMOKE_FILES+=("$file")
      ;;
  esac
done

if [[ "$DOCS_ONLY_OVERRIDE" == "1" || ( "$ALL_DOCS" == "1" && "$HAS_CODE_CHANGE" == "0" ) ]]; then
  echo "agent_validate_changed: docs-only change detected"
  echo "changed files:"
  printf '  %s\n' "${CHANGED_FILES[@]}"
  exit 0
fi

if [[ "${#PY_EXISTING_FILES[@]}" -gt 0 ]]; then
  run ./venv/bin/ruff check "${PY_EXISTING_FILES[@]}"
  run ./venv/bin/mypy "${PY_EXISTING_FILES[@]}"
  run ./venv/bin/python -m py_compile "${PY_EXISTING_FILES[@]}"
fi

if [[ "${#SHELL_FILES[@]}" -gt 0 ]]; then
  for file in "${SHELL_FILES[@]}"; do
    if path_exists "$file"; then
      run bash -n "$file"
    fi
  done
fi

if [[ "${#PY_EXISTING_FILES[@]}" -gt 0 ]]; then
  RUNTIME_PY_FILES=()
  NON_TEST_PY_FILES=()
  for file in "${PY_EXISTING_FILES[@]}"; do
    case "$file" in
      tests/*)
        ;;
      *)
        NON_TEST_PY_FILES+=("$file")
        ;;
    esac
    case "$file" in
      ai_trading/*.py|ai_trading/*/*.py|ai_trading/*/*/*.py)
        RUNTIME_PY_FILES+=("$file")
        ;;
    esac
  done
  if [[ "${#RUNTIME_PY_FILES[@]}" -gt 0 ]]; then
    if grep -nE 'except (Exception|BaseException)' "${RUNTIME_PY_FILES[@]}"; then
      echo "forbidden broad exception handler found in changed runtime Python files" >&2
      exit 1
    fi
    if grep -nE '(^|[^[:alnum:]_])print\(' "${RUNTIME_PY_FILES[@]}"; then
      echo "raw print found in changed runtime Python files" >&2
      exit 1
    fi
    if grep -nE 'subprocess\..*shell=True' "${RUNTIME_PY_FILES[@]}"; then
      echo "subprocess shell=True found in changed runtime Python files" >&2
      exit 1
    fi
    ENV_SCAN_FILES=()
    for file in "${RUNTIME_PY_FILES[@]}"; do
      case "$file" in
        ai_trading/config/management.py|ai_trading/config/*|ai_trading/settings.py)
          ;;
        *)
          ENV_SCAN_FILES+=("$file")
          ;;
      esac
    done
    if [[ "${#ENV_SCAN_FILES[@]}" -gt 0 ]] && grep -nE 'os\.(getenv|environ)' "${ENV_SCAN_FILES[@]}"; then
      echo "direct os environment access found outside approved config modules" >&2
      exit 1
    fi
  fi
  if [[ "${#NON_TEST_PY_FILES[@]}" -gt 0 ]] && grep -nE '(^|[^[:alnum:]_])pytz([^[:alnum:]_]|$)' "${NON_TEST_PY_FILES[@]}"; then
    echo "pytz reference found in changed Python files" >&2
    exit 1
  fi
  if [[ "${#NON_TEST_PY_FILES[@]}" -gt 0 ]] && grep -nE 'alpaca-trade-api|alpaca_trade_api' "${NON_TEST_PY_FILES[@]}"; then
    echo "legacy Alpaca SDK reference found in changed Python files" >&2
    exit 1
  fi
fi

if [[ "${#TEST_TARGETS[@]}" -gt 0 ]]; then
  mapfile -t SORTED_TEST_TARGETS < <(printf '%s\n' "${!TEST_TARGETS[@]}" | sort)
  run ./venv/bin/pytest -q "${SORTED_TEST_TARGETS[@]}"
elif [[ "$HAS_CODE_CHANGE" == "1" ]]; then
  echo "no related pytest target could be inferred for changed code; add a regression test or run --full" >&2
  exit 1
fi

if [[ "${#SYSTEMD_FILES[@]}" -gt 0 ]]; then
  if have_cmd systemd-analyze; then
    SYSTEMD_VERIFY_FILES=()
    for file in "${SYSTEMD_FILES[@]}"; do
      if path_exists "$file"; then
        SYSTEMD_VERIFY_FILES+=("$file")
      fi
    done
    if [[ "${#SYSTEMD_VERIFY_FILES[@]}" -gt 0 ]]; then
      run systemd-analyze verify "${SYSTEMD_VERIFY_FILES[@]}"
    fi
  else
    echo "systemd-analyze not available; cannot verify systemd unit changes" >&2
    exit 1
  fi
fi

if [[ "${#DEPENDENCY_FILES[@]}" -gt 0 ]]; then
  if [[ -x ./venv/bin/pip ]]; then
    run ./venv/bin/pip check
  else
    echo "./venv/bin/pip is unavailable; cannot run dependency consistency check" >&2
    exit 1
  fi
fi

if [[ "$SKIP_RUNTIME_SMOKE" != "1" && "${#RUNTIME_SMOKE_FILES[@]}" -gt 0 ]]; then
  run curl -sS http://127.0.0.1:9001/healthz
  run ./venv/bin/python -c "from tools import mcp_slack_alerts_server as s; payload=s.tool_runtime_incident_snapshot({}); assert payload.get('should_alert') is False, payload"
fi

echo "agent_validate_changed: targeted validation passed"
echo "changed files:"
printf '  %s\n' "${CHANGED_FILES[@]}"
