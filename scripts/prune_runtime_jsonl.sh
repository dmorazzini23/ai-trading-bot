#!/usr/bin/env bash
# Bounded retention for runtime JSONL evidence and research report runs.
#
# Live JSONL files above their byte cap are rotated: the live file keeps a
# bounded tail and the full pre-rotation file is archived, gzip-compressed,
# under the recovery backup tree. Research report run directories older than
# the archive age are packed into verified tar.gz archives in the same tree.
# Nothing is deleted: archives are never pruned by this script.
#
# Usage: prune_runtime_jsonl.sh [--dry-run|--apply]
#   --dry-run  print the retention plan and change nothing
#              (also AI_TRADING_RUNTIME_PRUNE_DRY_RUN=1)
set -euo pipefail
umask 077

DRY_RUN="${AI_TRADING_RUNTIME_PRUNE_DRY_RUN:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --apply) DRY_RUN=0 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done
[[ "$DRY_RUN" == "1" ]] || DRY_RUN=0

RUNTIME_DIR="${AI_TRADING_RUNTIME_DIR:-/var/lib/ai-trading-bot/runtime}"
DATA_DIR="${AI_TRADING_DATA_DIR:-$(dirname "$RUNTIME_DIR")}"
ARCHIVE_ROOT="${AI_TRADING_RUNTIME_ARCHIVE_DIR:-${RUNTIME_DIR}/recovery_backups/archive}"
JSONL_ARCHIVE_DIR="${ARCHIVE_ROOT}/jsonl"
RESEARCH_DIR="${AI_TRADING_RESEARCH_REPORTS_DIR:-${RUNTIME_DIR}/research_reports}"
RESEARCH_ARCHIVE_DIR="${ARCHIVE_ROOT}/research_reports"
RESEARCH_ARCHIVE_ENABLED="${AI_TRADING_RESEARCH_REPORT_ARCHIVE_ENABLED:-1}"
RESEARCH_ARCHIVE_DAYS="${AI_TRADING_RESEARCH_REPORT_ARCHIVE_DAYS:-30}"
RESEARCH_ARCHIVE_MAX_DIRS="${AI_TRADING_RESEARCH_REPORT_ARCHIVE_MAX_DIRS:-25}"
RESEARCH_ARCHIVE_SUBDIRS="${AI_TRADING_RESEARCH_REPORT_ARCHIVE_SUBDIRS:-daily weekly weekend monthly}"
SETTLE_SECONDS="${AI_TRADING_RUNTIME_PRUNE_SETTLE_SECONDS:-2}"
RUN_NAME_RE='^[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+$'

log() {
  printf '[runtime-prune] %s\n' "$*"
}

mode_label() {
  if [[ "$DRY_RUN" == "1" ]]; then echo "dry-run"; else echo "apply"; fi
}

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

for setting in RESEARCH_ARCHIVE_DAYS RESEARCH_ARCHIVE_MAX_DIRS SETTLE_SECONDS; do
  if ! is_uint "${!setting}"; then
    echo "invalid non-negative integer for ${setting}: ${!setting}" >&2
    exit 2
  fi
done

rotate_jsonl_file() {
  local path="$1"
  local max_bytes="$2"
  local keep_lines="$3"
  local name
  name="$(basename "$path")"

  if ! is_uint "$max_bytes" || ! is_uint "$keep_lines"; then
    echo "retention skipped for ${path}: invalid limits max_bytes=${max_bytes} keep_lines=${keep_lines}" >&2
    return 0
  fi
  if [[ ! -f "$path" || -L "$path" ]]; then
    [[ "$DRY_RUN" == "1" ]] && log "jsonl ${name}: action=none reason=missing"
    return 0
  fi
  [[ "$keep_lines" -gt 0 && "$max_bytes" -gt 0 ]] || return 0

  local size
  size="$(stat -c '%s' "$path" 2>/dev/null || echo 0)"
  if [[ "${size:-0}" -le "$max_bytes" ]]; then
    [[ "$DRY_RUN" == "1" ]] && log "jsonl ${name}: size=${size} max_bytes=${max_bytes} action=none reason=within_limit"
    return 0
  fi

  local ts target_bytes archive_dir archive tmp
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  target_bytes="$((max_bytes * 8 / 10))"
  archive_dir="${JSONL_ARCHIVE_DIR}/${name%.jsonl}"
  archive="${archive_dir}/${name}.${ts}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "jsonl ${name}: size=${size} max_bytes=${max_bytes} action=rotate keep_tail_bytes<=${target_bytes} keep_lines<=${keep_lines} archive=${archive}.gz"
    return 0
  fi

  mkdir -p "$archive_dir"
  local before_identity after_identity
  before_identity="$(stat -c '%i:%s:%Y' "$path")"
  tmp="${path}.tmp.${ts}"
  tail -c "$target_bytes" "$path" \
    | { IFS= read -r _partial_line || true; cat; } \
    | tail -n "$keep_lines" > "$tmp"
  after_identity="$(stat -c '%i:%s:%Y' "$path")"
  if [[ "$after_identity" != "$before_identity" ]]; then
    rm -f "$tmp"
    echo "retention deferred for ${path}: source changed during snapshot" >&2
    return 0
  fi
  # Hard-link the full file into the archive before replacing it, so the live
  # path never disappears and an in-flight append lands in the archived copy.
  if ! ln "$path" "$archive"; then
    rm -f "$tmp"
    echo "retention deferred for ${path}: archive must be on the runtime filesystem" >&2
    return 0
  fi
  chmod --reference="$path" "$tmp" 2>/dev/null || true
  chown --reference="$path" "$tmp" 2>/dev/null || true
  mv -f "$tmp" "$path"
  sleep "$SETTLE_SECONDS"
  if gzip "$archive" && gzip -t "${archive}.gz"; then
    log "jsonl ${name}: rotated size=${size} live_size=$(stat -c '%s' "$path") archive=${archive}.gz"
  else
    echo "archive compression failed for ${path}; uncompressed segment kept at ${archive}" >&2
  fi
}

collect_research_references() {
  # Run names referenced by model, runtime and recent report state. Referenced
  # runs stay in place because something may still open their files.
  local cutoff_epoch="$1"
  local pattern='[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_-]+'
  local sub entry
  {
    if [[ -d "${DATA_DIR}/models" ]]; then
      find "${DATA_DIR}/models" -type f -name '*.json' -print0 2>/dev/null
    fi
    find "$RUNTIME_DIR" -maxdepth 1 -type f -name '*.json' -print0 2>/dev/null
    find "$RUNTIME_DIR" -maxdepth 1 -type f -name '*.jsonl' -size -16M -print0 2>/dev/null
    find "$RESEARCH_DIR" -maxdepth 1 -type f -name '*.json' -print0 2>/dev/null
    for entry in "$RESEARCH_DIR"/*/; do
      entry="${entry%/}"
      sub="$(basename "$entry")"
      [[ "$sub" == "huggingface" ]] && continue
      if [[ " ${RESEARCH_ARCHIVE_SUBDIRS} " == *" ${sub} "* ]]; then
        find "$entry" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) \
          -newermt "@${cutoff_epoch}" -print0 2>/dev/null
      else
        find "$entry" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 2>/dev/null
      fi
    done
  } | xargs -0 -r grep -hoE "$pattern" 2>/dev/null | sort -u || true
}

archive_research_run() {
  local sub="$1"
  local run="$2"
  local dir="${RESEARCH_DIR}/${sub}/${run}"
  local dest_dir="${RESEARCH_ARCHIVE_DIR}/${sub}"
  local dest="${dest_dir}/${run}.tar.gz"
  local tmp="${dest_dir}/.${run}.tar.gz.tmp.$$"
  local nice_cmd=()

  [[ "$run" =~ $RUN_NAME_RE && -d "$dir" && ! -L "$dir" ]] || return 1
  if [[ -e "$dest" ]]; then
    echo "research archive exists for ${sub}/${run}; source left in place" >&2
    return 1
  fi
  mkdir -p "$dest_dir"
  command -v nice >/dev/null 2>&1 && nice_cmd+=(nice -n 19)
  command -v ionice >/dev/null 2>&1 && nice_cmd+=(ionice -c 3)

  local before after expected listed
  before="$(find "$dir" -printf '%s %T@ %p\n' | sort | sha256sum)"
  expected="$(find "$dir" | wc -l)"
  if ! "${nice_cmd[@]}" tar -C "${RESEARCH_DIR}/${sub}" -czf "$tmp" -- "$run"; then
    rm -f "$tmp"
    echo "research archive failed for ${sub}/${run}; source left in place" >&2
    return 1
  fi
  listed="$(tar -tzf "$tmp" | wc -l)"
  after="$(find "$dir" -printf '%s %T@ %p\n' | sort | sha256sum)"
  if [[ "$listed" != "$expected" || "$before" != "$after" ]] || ! gzip -t "$tmp"; then
    rm -f "$tmp"
    echo "research archive verification failed for ${sub}/${run} (entries ${listed}/${expected}); source left in place" >&2
    return 1
  fi
  mv "$tmp" "$dest"
  sync "$dest" 2>/dev/null || true
  rm -rf -- "${RESEARCH_DIR:?}/${sub:?}/${run:?}"
  log "research_reports ${sub}/${run}: archived entries=${expected} archive=${dest}"
}

archive_research_reports() {
  if [[ "$RESEARCH_ARCHIVE_ENABLED" != "1" ]]; then
    log "research_reports: action=none reason=disabled"
    return 0
  fi
  if [[ ! -d "$RESEARCH_DIR" || -L "$RESEARCH_DIR" ]]; then
    [[ "$DRY_RUN" == "1" ]] && log "research_reports: action=none reason=missing dir=${RESEARCH_DIR}"
    return 0
  fi

  local now_epoch cutoff_epoch cutoff_stamp
  now_epoch="$(date -u +%s)"
  cutoff_epoch="$((now_epoch - RESEARCH_ARCHIVE_DAYS * 86400))"
  cutoff_stamp="$(date -u -d "@${cutoff_epoch}" +%Y%m%dT%H%M%SZ)"

  local sub entry run stamp
  local -a candidates=()
  local recent_runs=0 recent_files=0
  for sub in $RESEARCH_ARCHIVE_SUBDIRS; do
    [[ "$sub" =~ ^[A-Za-z0-9_-]+$ ]] || continue
    [[ -d "${RESEARCH_DIR}/${sub}" && ! -L "${RESEARCH_DIR}/${sub}" ]] || continue
    for entry in "${RESEARCH_DIR}/${sub}"/*/; do
      [[ -d "$entry" ]] || continue
      entry="${entry%/}"
      run="$(basename "$entry")"
      [[ "$run" =~ $RUN_NAME_RE && ! -L "$entry" ]] || continue
      stamp="${run%%_*}"
      if [[ ! "$stamp" < "$cutoff_stamp" ]]; then
        recent_runs=$((recent_runs + 1))
        continue
      fi
      if [[ -n "$(find "$entry" -newermt "@${cutoff_epoch}" -print -quit 2>/dev/null)" ]]; then
        recent_files=$((recent_files + 1))
        [[ "$DRY_RUN" == "1" ]] && log "research_reports ${sub}/${run}: action=keep reason=files_newer_than_${RESEARCH_ARCHIVE_DAYS}d"
        continue
      fi
      candidates+=("${sub}/${run}")
    done
  done

  local not_managed=""
  for entry in "$RESEARCH_DIR"/*/; do
    [[ -d "$entry" ]] || continue
    sub="$(basename "${entry%/}")"
    [[ " ${RESEARCH_ARCHIVE_SUBDIRS} " == *" ${sub} "* ]] || not_managed="${not_managed:+${not_managed},}${sub}"
  done

  local references=""
  if [[ "${#candidates[@]}" -gt 0 ]]; then
    references="$(collect_research_references "$cutoff_epoch")"
  fi

  local candidate bytes archived=0 pinned=0 planned=0 deferred=0
  local planned_bytes=0 pinned_bytes=0
  for candidate in "${candidates[@]}"; do
    sub="${candidate%%/*}"
    run="${candidate#*/}"
    bytes="$(du -sb "${RESEARCH_DIR}/${candidate}" 2>/dev/null | cut -f1)"
    bytes="${bytes:-0}"
    if [[ -n "$references" ]] && grep -qxF -- "$run" <<<"$references"; then
      pinned=$((pinned + 1))
      pinned_bytes=$((pinned_bytes + bytes))
      [[ "$DRY_RUN" == "1" ]] && log "research_reports ${candidate}: bytes=${bytes} action=keep reason=referenced_by_model_or_runtime_state"
      continue
    fi
    planned=$((planned + 1))
    planned_bytes=$((planned_bytes + bytes))
    if [[ "$DRY_RUN" == "1" ]]; then
      log "research_reports ${candidate}: bytes=${bytes} action=archive archive=${RESEARCH_ARCHIVE_DIR}/${candidate}.tar.gz"
      continue
    fi
    if [[ "$RESEARCH_ARCHIVE_MAX_DIRS" -gt 0 && "$archived" -ge "$RESEARCH_ARCHIVE_MAX_DIRS" ]]; then
      deferred=$((deferred + 1))
      continue
    fi
    if archive_research_run "$sub" "$run"; then
      archived=$((archived + 1))
    fi
  done

  log "research_reports summary: mode=$(mode_label) cutoff=${cutoff_stamp} archive_candidates=${planned} candidate_bytes=${planned_bytes} archived=${archived} deferred_to_next_run=${deferred} kept_referenced=${pinned} referenced_bytes=${pinned_bytes} kept_recent_runs=${recent_runs} kept_recent_files=${recent_files} per_run_cap=${RESEARCH_ARCHIVE_MAX_DIRS} not_managed=${not_managed:-none}"
}

report_legacy_backups() {
  # Older in-place backups are no longer auto-deleted; list them for operators.
  [[ "$DRY_RUN" == "1" ]] || return 0
  local legacy
  for legacy in "$RUNTIME_DIR"/*.jsonl.bak* "$RUNTIME_DIR"/*.jsonl.*.bak; do
    [[ -f "$legacy" ]] || continue
    log "legacy_backup $(basename "$legacy"): bytes=$(stat -c '%s' "$legacy") mtime=$(date -u -d "@$(stat -c '%Y' "$legacy")" +%Y-%m-%d) action=none reason=kept_in_place"
  done
}

log "start mode=$(mode_label) runtime_dir=${RUNTIME_DIR} archive_root=${ARCHIVE_ROOT}"

# Per-file caps keep the live file bounded while preserving each reader's
# window; see docs/OPERATIONS.md "Runtime JSONL retention".
rotate_jsonl_file \
  "${RUNTIME_DIR}/decision_records.jsonl" \
  "${AI_TRADING_DECISION_RECORDS_MAX_BYTES:-201326592}" \
  "${AI_TRADING_DECISION_RECORDS_KEEP_LINES:-50000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/config_snapshots.jsonl" \
  "${AI_TRADING_CONFIG_SNAPSHOTS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_CONFIG_SNAPSHOTS_KEEP_LINES:-30000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/gate_effectiveness.jsonl" \
  "${AI_TRADING_GATE_EFFECTIVENESS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_GATE_EFFECTIVENESS_KEEP_LINES:-300000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/ml_shadow_predictions.jsonl" \
  "${AI_TRADING_ML_SHADOW_PREDICTIONS_MAX_BYTES:-268435456}" \
  "${AI_TRADING_ML_SHADOW_PREDICTIONS_KEEP_LINES:-100000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/execution_quality_events.jsonl" \
  "${AI_TRADING_EXECUTION_QUALITY_EVENTS_MAX_BYTES:-67108864}" \
  "${AI_TRADING_EXECUTION_QUALITY_EVENTS_KEEP_LINES:-100000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/broker_position_boundaries.jsonl" \
  "${AI_TRADING_BROKER_POSITION_BOUNDARIES_MAX_BYTES:-33554432}" \
  "${AI_TRADING_BROKER_POSITION_BOUNDARIES_KEEP_LINES:-150000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/policy_ablation_events.jsonl" \
  "${AI_TRADING_POLICY_ABLATION_EVENTS_MAX_BYTES:-33554432}" \
  "${AI_TRADING_POLICY_ABLATION_EVENTS_KEEP_LINES:-100000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/order_events.jsonl" \
  "${AI_TRADING_ORDER_EVENTS_MAX_BYTES:-134217728}" \
  "${AI_TRADING_ORDER_EVENTS_KEEP_LINES:-150000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/fill_events.jsonl" \
  "${AI_TRADING_FILL_EVENTS_MAX_BYTES:-134217728}" \
  "${AI_TRADING_FILL_EVENTS_KEEP_LINES:-150000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/tca_records.jsonl" \
  "${AI_TRADING_TCA_RECORDS_MAX_BYTES:-134217728}" \
  "${AI_TRADING_TCA_RECORDS_KEEP_LINES:-150000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/oms_events.jsonl" \
  "${AI_TRADING_OMS_EVENTS_MAX_BYTES:-67108864}" \
  "${AI_TRADING_OMS_EVENTS_KEEP_LINES:-100000}"

rotate_jsonl_file \
  "${RUNTIME_DIR}/memory_samples.jsonl" \
  "${AI_TRADING_MEMORY_SAMPLES_MAX_BYTES:-5000000}" \
  "${AI_TRADING_MEMORY_SAMPLES_KEEP_LINES:-20000}"

report_legacy_backups
archive_research_reports
log "done mode=$(mode_label)"
