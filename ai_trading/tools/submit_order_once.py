from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from ai_trading.alpaca_api import alpaca_get, submit_order
from ai_trading.exc import HTTPError
from ai_trading.logging import get_logger
from ai_trading.utils.time import monotonic_time


logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a single order and optionally poll status")
    parser.add_argument("--dotenv", default=None, help="Path to .env to load (optional)")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--side", required=True, choices=["buy", "sell"], help="Order side")
    parser.add_argument("--qty", required=True, type=float, help="Quantity to submit")
    parser.add_argument("--type", default="market", choices=["market", "limit", "stop", "stop_limit"], help="Order type")
    parser.add_argument("--time-in-force", dest="tif", default="day", choices=["day", "gtc", "opg", "cls", "ioc", "fok"], help="Time in force")
    parser.add_argument("--limit-price", type=float, default=None, help="Limit price for limit/stop_limit orders")
    parser.add_argument("--stop-price", type=float, default=None, help="Stop price for stop/stop_limit orders")
    parser.add_argument("--idempotency-key", default=None, help="Explicit client order id (optional)")
    parser.add_argument("--shadow", dest="shadow", action="store_true", help="Force shadow mode")
    parser.add_argument("--no-shadow", dest="shadow", action="store_false", help="Force non-shadow submit")
    parser.set_defaults(shadow=None)
    parser.add_argument("--poll/--no-poll", dest="poll", action=argparse.BooleanOptionalAction, default=True, help="Poll the order status")
    parser.add_argument("--poll-every", type=float, default=0.5, help="Polling interval seconds")
    parser.add_argument("--poll-timeout", type=float, default=15.0, help="Max seconds to poll before giving up")
    return parser.parse_args(argv)


def _is_terminal(status: str) -> bool:
    s = (status or "").strip().lower()
    return s in {"filled", "canceled", "cancelled", "rejected", "expired"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Load environment early so config resolves
    try:
        from ai_trading.env import ensure_dotenv_loaded

        ensure_dotenv_loaded(args.dotenv)
    except Exception:
        # Continue; downstream calls will still function with process env
        logger.debug("DOTENV_LOAD_FAILED", exc_info=True)

    logger.info(
        "ORDER_SUBMIT_REQUEST",
        extra={
            "symbol": args.symbol,
            "side": args.side,
            "qty": args.qty,
            "type": args.type,
            "tif": args.tif,
            "limit_price": args.limit_price,
            "stop_price": args.stop_price,
            "shadow_override": args.shadow,
            "poll": args.poll,
            "poll_every": args.poll_every,
            "poll_timeout": args.poll_timeout,
        },
    )

    try:
        order: dict[str, Any] = submit_order(
            args.symbol,
            args.side,
            qty=args.qty,
            type=args.type,
            time_in_force=args.tif,
            limit_price=args.limit_price,
            stop_price=args.stop_price,
            shadow=args.shadow,
            idempotency_key=args.idempotency_key,
            timeout=10,
        )
    except HTTPError as e:
        logger.error("ORDER_SUBMIT_HTTP_ERROR", extra={"error": str(e)})
        return 2
    except Exception as e:  # pragma: no cover - defensive
        logger.error("ORDER_SUBMIT_ERROR", extra={"error": str(e)})
        return 1

    order_id = str(order.get("id") or "")
    logger.info("ORDER_SUBMIT_ACCEPTED", extra={"order_id": order_id, "status": order.get("status")})

    # In shadow mode, returned IDs are synthetic; skip HTTP polling
    is_shadow_id = order_id.startswith("shadow-")
    if not args.poll or not order_id or is_shadow_id:
        return 0

    # Poll for terminal state
    start = monotonic_time()
    last_payload: dict[str, Any] | None = None
    while monotonic_time() - start < float(args.poll_timeout):
        try:
            payload = alpaca_get(f"/v2/orders/{order_id}")
        except HTTPError as e:  # pragma: no cover - network error path
            logger.warning("ORDER_POLL_HTTP_ERROR", extra={"order_id": order_id, "error": str(e)})
            time.sleep(max(0.05, float(args.poll_every)))
            continue
        status = str(payload.get("status") or "")
        last_payload = payload if isinstance(payload, dict) else None
        logger.info("ORDER_POLL", extra={"order_id": order_id, "status": status})
        if _is_terminal(status):
            logger.info("ORDER_FINAL", extra={"order_id": order_id, "status": status, "payload": last_payload})
            return 0
        time.sleep(max(0.05, float(args.poll_every)))

    logger.warning("ORDER_POLL_TIMEOUT", extra={"order_id": order_id, "timeout": args.poll_timeout, "payload": last_payload})
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
