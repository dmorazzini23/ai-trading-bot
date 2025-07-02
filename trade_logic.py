# AI-AGENT-REF: basic trade utilities

def compute_order_price(symbol_data):
    raw_price = extract_price(symbol_data)
    return max(raw_price, 1e-3)  # avoid non-positive prices
