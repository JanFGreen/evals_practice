# USD per 1M tokens, Anthropic first-party API rates as of 2026-11. Verify before relying on them.
PRICES = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25


def cost_usd(model: str, usage: dict) -> float | None:
    price = PRICES.get(model)
    if price is None or not usage:
        return None
    p_in, p_out = price[0] / 1e6, price[1] / 1e6
    return (
        usage.get("input_tokens", 0) * p_in
        + usage.get("output_tokens", 0) * p_out
        + usage.get("cache_read_input_tokens", 0) * p_in * CACHE_READ_MULT
        + usage.get("cache_creation_input_tokens", 0) * p_in * CACHE_WRITE_MULT
    )
