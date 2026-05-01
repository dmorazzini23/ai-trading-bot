from __future__ import annotations

from ai_trading.config.operator_auth import parse_operator_token_map


def test_parse_operator_token_map_accepts_json_and_fallback() -> None:
    assert parse_operator_token_map('{"Alice": "secret"}') == {"alice": "secret"}
    assert parse_operator_token_map("Bob=token, Carol:other, broken") == {
        "bob": "token",
        "carol": "other",
    }


def test_parse_operator_token_map_ignores_empty_entries() -> None:
    assert parse_operator_token_map({"Alice": "secret"}) == {"alice": "secret"}
    assert parse_operator_token_map('{"Alice": "", "": "secret"}') == {}
