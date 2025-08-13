import logging
import time
import types
import uuid

# AI-AGENT-REF: lightweight Alpaca API stub for tests
SHADOW_MODE = False
DRY_RUN = False
requests = types.SimpleNamespace(exceptions=types.SimpleNamespace(HTTPError=Exception, RequestException=Exception))
partial_fill_tracker = {}
partial_fills = []

def submit_order(api, order_data, log=None):
    if SHADOW_MODE:
        return {"status": "shadow"}
    if DRY_RUN:
        return {"status": "dry_run"}
    if getattr(order_data, "client_order_id", None) is None:
        setattr(order_data, "client_order_id", str(uuid.uuid4()))
    while True:
        resp = api.submit_order(order_data)
        if getattr(resp, "status_code", 200) == 429:
            time.sleep(1)
            continue
        return resp

async def handle_trade_update(event):
    logger = logging.getLogger(__name__)
    oid = getattr(event.order, "id", None)
    if event.event == "partial_fill":
        if oid in partial_fill_tracker:
            return
        partial_fill_tracker[oid] = event.order.filled_qty
        partial_fills.append(oid)
        logger.debug("ORDER_PARTIAL_FILL")
    elif event.event == "fill":
        partial_fill_tracker.pop(oid, None)
        logger.debug("ORDER_FILLED")
