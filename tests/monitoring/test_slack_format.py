from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_trading.monitoring import slack_format as fmt


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-3.9400142214482634, "-$3.94"), (0.76, "+$0.76"), (0, "$0.00"), (1234.5, "+$1,234.50"), (None, "n/a")],
)
def test_fmt_money_puts_sign_before_symbol(value: object, expected: str) -> None:
    assert fmt.fmt_money(value) == expected


def test_number_formats_round_and_label_units() -> None:
    assert fmt.fmt_bps(8.384140428738098) == "8.4 bps"
    assert fmt.fmt_ratio(0.3150917379726395) == "0.32"
    assert fmt.fmt_pct(0.0889) == "8.9%"
    assert fmt.fmt_pct(1.25, digits=2, ratio=False) == "1.25%"
    assert fmt.fmt_count(266704) == "266,704"
    assert fmt.fmt_count(True) == "n/a"
    assert fmt.fmt_bps(float("nan")) == "n/a"
    assert [fmt.fmt_age(s) for s in (45, 720, 12600, 200000, None)] == ["45s", "12m", "3.5h", "2.3d", "n/a"]


def test_times_render_in_eastern() -> None:
    assert fmt.fmt_time("2026-10-16T19:55:35.000083+00:00") == "Fri Oct 16, 3:55 PM ET"
    assert fmt.fmt_time("2026-10-18T05:01:36.829909Z", with_date=False) == "1:01 AM ET"
    assert fmt.fmt_time(datetime(2026, 1, 5, 14, 30, tzinfo=UTC)) == "Mon Jan 5, 9:30 AM ET"
    assert fmt.fmt_time("not-a-time") == "not-a-time"
    assert fmt.fmt_time("") == "n/a"
    assert fmt.fmt_day("2026-10-16") == "Fri Oct 16"


def test_humanize_turns_codes_into_words() -> None:
    assert fmt.humanize("required_model_stale") == "required model stale"
    assert fmt.humanize("LONG_ONLY_SHORT_SUPPRESSED") == "long only short suppressed"
    assert fmt.humanize("alpaca") == "alpaca"
    assert fmt.humanize(None) == "n/a"
    assert fmt.capitalized("warming_up") == "Warming up"


def test_lists_truncate_with_count() -> None:
    items = [f"item {index}" for index in range(8)]
    assert fmt.bullets(items).splitlines() == [
        "• item 0", "• item 1", "• item 2", "• item 3", "• item 4", "• …and 3 more",
    ]
    assert fmt.bullets([]) == "• none"
    assert fmt.inline_list(items, limit=2) == "item 0, item 1 (+6 more)"
    assert fmt.inline_list(["", None]) == "none"


def test_blocks_respect_slack_limits() -> None:
    header = fmt.header_block(fmt.Severity.CRITICAL, "x" * 400)
    assert header["text"]["type"] == "plain_text"
    assert header["text"]["text"].startswith("🔴 ")
    assert len(header["text"]["text"]) == fmt.HEADER_MAX_CHARS
    assert len(fmt.section_block("y" * 5000)["text"]["text"]) == fmt.SECTION_MAX_CHARS
    fields = fmt.fields_block([(f"L{i}", i) for i in range(14)], title="Impact")
    assert len(fields["fields"]) == 10
    assert fields["fields"][0] == {"type": "mrkdwn", "text": "*L0*\n0"}
    assert fields["text"] == {"type": "mrkdwn", "text": "*Impact*"}
    context = fmt.context_block("a" * 400, "", "b")
    assert [len(element["text"]) for element in context["elements"]] == [fmt.CONTEXT_MAX_CHARS, 1]


def test_message_drops_empty_context_and_builds_fallback() -> None:
    text = fmt.fallback_text(fmt.Severity.WARNING, "Runtime health is degraded", "health: stale", "")
    payload = fmt.message(text, [fmt.header_block(fmt.Severity.WARNING, "t"), fmt.context_block("")])
    assert payload == {
        "text": "[WARNING] Runtime health is degraded — health: stale",
        "blocks": [fmt.header_block(fmt.Severity.WARNING, "t")],
    }
    assert [severity.emoji for severity in fmt.Severity] == ["🔴", "🟠", "🟢", "🔵"]
