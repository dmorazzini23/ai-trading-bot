"""Shared Slack message formatting: severity markers, value formats and Block Kit builders.

Every Slack sender in the repo renders through these helpers so messages share
one layout: a header with a severity marker, labeled values with units, short
bullet lists that end with a "…and N more" count, and a grey context line.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")

HEADER_MAX_CHARS = 150
SECTION_MAX_CHARS = 3000
FIELD_MAX_CHARS = 2000
CONTEXT_MAX_CHARS = 300
MAX_FIELDS = 10
MAX_CONTEXT_ELEMENTS = 10
LIST_LIMIT = 5


class Severity(Enum):
    """Severity marker shown in the header and the notification preview."""

    CRITICAL = ("🔴", "CRITICAL")
    WARNING = ("🟠", "WARNING")
    OK = ("🟢", "OK")
    INFO = ("🔵", "INFO")

    @property
    def emoji(self) -> str:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def fmt_money(value: Any, *, signed: bool = True) -> str:
    """Dollars with the sign before the symbol: '-$3.94', '+$0.76'."""

    number = _number(value)
    if number is None:
        return "n/a"
    if number < 0:
        sign = "-"
    elif signed and number > 0:
        sign = "+"
    else:
        sign = ""
    return f"{sign}${abs(number):,.2f}"


def fmt_bps(value: Any, *, digits: int = 1) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:,.{digits}f} bps"


def fmt_ratio(value: Any, *, digits: int = 2) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def fmt_pct(value: Any, *, digits: int = 1, ratio: bool = True) -> str:
    """Percent; ``ratio=True`` means the input is a 0..1 fraction."""

    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number * 100.0 if ratio else number:.{digits}f}%"


def fmt_count(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{int(number):,}"


def fmt_age(seconds: Any) -> str:
    """Compact age: '45s', '12m', '3.5h', '2.1d'."""

    number = _number(seconds)
    if number is None:
        return "n/a"
    if number < 120:
        return f"{number:.0f}s"
    if number < 7200:
        return f"{number / 60:.0f}m"
    if number < 172800:
        return f"{number / 3600:.1f}h"
    return f"{number / 86400:.1f}d"


def fmt_time(value: Any, *, with_date: bool = True) -> str:
    """Eastern wall time, e.g. 'Fri Oct 16, 3:55 PM ET'; unparseable text is returned as-is."""

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return "n/a"
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(_EASTERN)
    clock = local.strftime("%I:%M %p").lstrip("0")
    if not with_date:
        return f"{clock} ET"
    return f"{local.strftime('%a %b')} {local.day}, {clock} ET"


def fmt_day(value: Any) -> str:
    """A YYYY-MM-DD date as 'Fri Oct 16'."""

    text = str(value or "").strip()
    try:
        parsed = datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return text or "n/a"
    return f"{parsed.strftime('%a %b')} {parsed.day}"


def now_label() -> str:
    return fmt_time(datetime.now(UTC))


def humanize(code: Any) -> str:
    """Internal codes as words: 'required_model_stale' -> 'required model stale'."""

    text = str(code if code not in (None, "") else "n/a").strip()
    if "_" not in text and not text.isupper():
        return text
    return text.replace("_", " ").lower()


def capitalized(code: Any) -> str:
    text = humanize(code)
    return text[:1].upper() + text[1:]


def yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def truncate(text: Any, max_len: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


def _limited(items: Iterable[Any], limit: int) -> tuple[list[str], int]:
    rows = [str(item).strip() for item in items if str(item or "").strip()]
    return rows[:limit], max(len(rows) - limit, 0)


def bullets(items: Iterable[Any], *, limit: int = LIST_LIMIT, empty: str = "• none") -> str:
    """Bullet lines, capped at ``limit`` with a trailing '…and N more'."""

    shown, hidden = _limited(items, limit)
    if not shown:
        return empty
    lines = [f"• {item}" for item in shown]
    if hidden:
        lines.append(f"• …and {hidden} more")
    return "\n".join(lines)


def inline_list(items: Iterable[Any], *, limit: int = LIST_LIMIT, empty: str = "none") -> str:
    """Comma list, capped at ``limit`` with a trailing '(+N more)'."""

    shown, hidden = _limited(items, limit)
    if not shown:
        return empty
    text = ", ".join(shown)
    return f"{text} (+{hidden} more)" if hidden else text


def headline(severity: Severity, title: str) -> str:
    return truncate(f"{severity.emoji} {title}", HEADER_MAX_CHARS)


def fallback_text(severity: Severity, title: str, *details: str) -> str:
    """Notification preview: '[WARNING] title — detail · detail'."""

    tail = " · ".join(item for item in details if item)
    text = f"[{severity.label}] {title}"
    return truncate(f"{text} — {tail}" if tail else text, SECTION_MAX_CHARS)


def header_block(severity: Severity, title: str) -> dict[str, Any]:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": headline(severity, title), "emoji": True},
    }


def section_block(text: str) -> dict[str, Any]:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": truncate(text, SECTION_MAX_CHARS) or "n/a"},
    }


def fields_block(pairs: Sequence[tuple[str, Any]], *, title: str | None = None) -> dict[str, Any]:
    """A section of up to ten '*label*\\nvalue' fields, optionally titled."""

    block: dict[str, Any] = {
        "type": "section",
        "fields": [
            {"type": "mrkdwn", "text": truncate(f"*{label}*\n{value}", FIELD_MAX_CHARS)}
            for label, value in list(pairs)[:MAX_FIELDS]
        ],
    }
    if title:
        block["text"] = {"type": "mrkdwn", "text": f"*{title}*"}
    return block


def context_block(*items: str) -> dict[str, Any]:
    elements = [
        {"type": "mrkdwn", "text": truncate(item, CONTEXT_MAX_CHARS)} for item in items if item
    ]
    return {"type": "context", "elements": elements[:MAX_CONTEXT_ELEMENTS]}


def message(text: str, blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Assemble a webhook payload, dropping context blocks that ended up empty."""

    kept = [block for block in blocks if block.get("type") != "context" or block.get("elements")]
    return {"text": text, "blocks": kept}
