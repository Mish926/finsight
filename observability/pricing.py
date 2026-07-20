"""Model pricing for cost estimation.

Prices are USD per 1M tokens (input, output). The table ships with common
Groq and OpenAI models; anything else can be registered at runtime with
register_pricing(). Unknown models cost $0 and are reported as unpriced
rather than silently guessed.

Prices drift over time. Treat these as estimates and re-verify against the
provider's pricing page before quoting numbers anywhere that matters.
"""

from __future__ import annotations

# model name -> (usd per 1M input tokens, usd per 1M output tokens)
_PRICING: dict[str, tuple[float, float]] = {
    # Groq
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

_unpriced_models: set[str] = set()


def register_pricing(model: str, input_per_1m: float, output_per_1m: float) -> None:
    """Register or override pricing for a model."""
    _PRICING[model] = (float(input_per_1m), float(output_per_1m))
    _unpriced_models.discard(model)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost of a call. Unknown models return 0.0 and are
    tracked so reports can flag them instead of hiding the gap."""
    pricing = _PRICING.get(model)
    if pricing is None:
        # Try prefix match: providers often version model names
        # (e.g. "gpt-4o-2024-08-06").
        for known, p in _PRICING.items():
            if model.startswith(known):
                pricing = p
                break
    if pricing is None:
        _unpriced_models.add(model)
        return 0.0
    in_price, out_price = pricing
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def unpriced_models() -> set[str]:
    """Models seen during this session that had no pricing entry."""
    return set(_unpriced_models)
